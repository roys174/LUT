"""Hard (original, pre-new-attention) attention in lut_torch.

``test_torch_parity.py`` pins ``--attention hard`` to ``main.py`` wherever ``main.py`` has
the structure. This file covers what parity cannot:

  * the gated-group primitive against the pure-autograd oracle, and against an explicit
    transcription of ``main._attention_backward_kernel``'s gating rule;
  * the torch-only extensions (``--new-MHA``, ``--weight-sharing``, ``--hard-attn-n-c``),
    which ``main.py`` never had, against the oracle at whole-model level;
  * configuration validation and trainer wiring.

Run with:  .venv/bin/python test_torch_hard_attention.py
       or: python -m pytest -q test_torch_hard_attention.py
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

import lut_torch.modules as modules
from lut_torch._testing import parametrize, run_all
from lut_torch.config import ConfigError, LutConfig
from lut_torch.functional import (
    GATE_CONDITIONAL,
    GATE_EXCLUSIVE,
    LutGroup,
    lut_lookup,
    lut_lookup_reference,
)
from lut_torch.modules import HardAttentionHead, LutModel
from lut_torch.train import build_parser

BASE = dict(
    context_size=8, embedding_dim=16, positional_dim=3, num_layers=2, num_heads=2,
    vocab_size_input=23, vocab_size_output=7, n_t=4, n_c=3, POS_task=True,
    attention='hard',
)


def make_model(seed=3, **overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    cfg = LutConfig(**kwargs)
    random.seed(seed)
    np.random.seed(seed)
    model = LutModel(cfg)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed)
        for table in model.table_parameters():
            table.copy_(torch.randn(table.shape, generator=gen) * 0.3)
    return model, cfg


def hard_call(seed, batch=2, cs=6, n_t=3, n_bits=2, fan_in=1, pd=3, y_dim=5, share=1):
    """Random inputs for one hard-attention-shaped gated lookup."""
    gen = torch.Generator().manual_seed(seed)
    tok_i, tok_j = modules.causal_pair_indices(cs)
    tok_i, tok_j = torch.tensor(tok_i), torch.tensor(tok_j)
    pe_pos = tok_i - tok_j
    total = 2 * n_bits + pd
    s_n_t = n_t // share
    S = (torch.randn(s_n_t * (1 << total), y_dim, generator=gen) * 0.5).requires_grad_()
    trees = torch.arange(s_n_t).repeat_interleave(share)
    qk = torch.randn(batch, cs, n_t, n_bits * fan_in, generator=gen).requires_grad_()
    pe = torch.randn(cs, n_t, pd, generator=gen).requires_grad_()
    groups = [
        LutGroup(n_bits, fan_in, pd, pair_index=tok_j, gate=GATE_EXCLUSIVE),
        LutGroup(n_bits, fan_in, n_bits + pd, pair_index=tok_i, gate=GATE_EXCLUSIVE),
        LutGroup(pd, 1, 0, pair_index=pe_pos, gate=GATE_CONDITIONAL),
    ]
    grad_y = torch.randn(batch, len(tok_i), y_dim, generator=gen)
    return S, trees, 1 << total, groups, qk, pe, grad_y


def run_call(fn, S, trees, n_bins, groups, qk, pe, grad_y, **kwargs):
    for t in (S, qk, pe):
        t.grad = None
    margins = [qk, qk, pe.unsqueeze(0).expand(qk.shape[0], -1, -1, -1)]
    y = fn(S, trees, n_bins, groups, margins, **kwargs)
    (y * grad_y).sum().backward()
    S_grad = S.grad.to_dense() if S.grad.is_sparse else S.grad
    return y.detach(), S_grad.clone(), qk.grad.clone(), pe.grad.clone()


# ---------------------------------------------------------------------------
# The gated primitive
# ---------------------------------------------------------------------------

@parametrize('fan_in', [1, 2])
@parametrize('share', [1, 3])
@parametrize('table_grad', ['dense', 'sparse'])
def test_gated_function_matches_reference(fan_in, share, table_grad):
    call = hard_call(seed=fan_in * 10 + share, fan_in=fan_in, share=share)
    got = run_call(lut_lookup, *call, table_grad=table_grad)
    want = run_call(lut_lookup_reference, *call)
    for g, w, label in zip(got, want, ('y', 'S', 'qk', 'pe')):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6, msg=label)


def test_gating_matches_main_rule_by_hand():
    """Transcribe main._attention_backward_kernel's Phase 1 and compare margin grads."""
    S, trees, n_bins, groups, qk, pe, grad_y = hard_call(seed=7, batch=1, n_t=2)
    _, _, got_qk, got_pe = run_call(lut_lookup, S, trees, n_bins, groups, qk, pe, grad_y)

    n_bits, pd = groups[0].n_bits, groups[2].n_bits
    shift_qk = n_bits + pd
    Sv = S.detach().view(-1, n_bins, S.shape[-1])
    u_v, u_pe = qk.detach()[0], pe.detach()
    r_v, r_pe = u_v.abs().argmin(-1), u_pe.abs().argmin(-1)
    j_v = ((u_v > 0).long() << torch.arange(n_bits)).sum(-1)
    j_pe = ((u_pe > 0).long() << torch.arange(pd)).sum(-1)
    want_qk = torch.zeros_like(u_v)
    want_pe = torch.zeros_like(u_pe)
    surrogate = lambda u: -0.5 * (1.0 if u > 0 else -1.0) / (1.0 + abs(u)) ** 2

    cs, n_t = u_v.shape[:2]
    pair = 0
    for pos in range(1, cs):
        for pos1 in range(pos):
            g = grad_y[0, pair]
            pair += 1
            for t in range(n_t):
                tree = int(trees[t])
                j = (int(j_v[pos, t]) << shift_qk) | (int(j_v[pos1, t]) << pd) \
                    | int(j_pe[pos - pos1, t])
                uq = float(u_v[pos, t, r_v[pos, t]])
                uk = float(u_v[pos1, t, r_v[pos1, t]])
                q_wins = abs(uq) < abs(uk)
                src, u = (pos, uq) if q_wins else (pos1, uk)
                r = int(r_v[src, t])
                flip = (r // groups[0].fan_in) + (shift_qk if q_wins else pd)
                gi = float(((Sv[tree, j ^ (1 << flip)] - Sv[tree, j]) * g).sum())
                want_qk[src, t, r] += gi * surrogate(u)

                upe = float(u_pe[pos - pos1, t, r_pe[pos - pos1, t]])
                if abs(upe) < abs(uq) and abs(upe) < abs(uk):
                    rp = int(r_pe[pos - pos1, t])
                    gi = float(((Sv[tree, j ^ (1 << rp)] - Sv[tree, j]) * g).sum())
                    want_pe[pos - pos1, t, rp] += gi * surrogate(upe)

    torch.testing.assert_close(got_qk[0], want_qk, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(got_pe, want_pe, rtol=1e-4, atol=1e-6)
    # The rule must actually have done something in both directions.
    assert (want_pe != 0).any() and (want_pe == 0).sum() > 0


def test_tie_goes_to_the_key():
    """main.py flips the query only on a strict |uQ| < |uK|; a tie flips the key."""
    S, trees, n_bins, groups, qk, pe, grad_y = hard_call(seed=1, batch=1, cs=2, n_t=1,
                                                         n_bits=1)
    with torch.no_grad():
        qk.copy_(torch.tensor([[[[0.4]]], [[[-0.4]]]]).reshape_as(qk))
        pe.fill_(5.0)                  # never the closest -> never gated in
    _, _, got_qk, got_pe = run_call(lut_lookup, S, trees, n_bins, groups, qk, pe, grad_y)
    assert got_qk[0, 1, 0, 0] == 0          # query (position 1) gets nothing
    assert got_qk[0, 0, 0, 0] != 0          # key (position 0) gets the gradient
    assert (got_pe == 0).all()


def test_gated_rejects_cosine():
    S, trees, n_bins, groups, qk, pe, _ = hard_call(seed=2)
    try:
        lut_lookup(S, trees, n_bins, groups, [qk, qk, pe.unsqueeze(0).expand(2, -1, -1, -1)],
                   selection='cosine-distance')
    except NotImplementedError:
        return
    raise AssertionError('cosine-distance with gated groups must be refused')


# ---------------------------------------------------------------------------
# Whole model, including the torch-only extensions
# ---------------------------------------------------------------------------

def _reference_lookup(S_flat, trees, n_bins, groups, margins, selection=None,
                      table_grad=None, capture=None, tree_weights=None):
    return lut_lookup_reference(S_flat, trees, n_bins, groups, margins, capture=capture,
                                tree_weights=tree_weights)


def _model_grads(model, cfg, tokens, targets):
    model.zero_grad(set_to_none=True)
    logits = model(tokens)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size_output), targets.reshape(-1),
                    reduction='sum').backward()
    out = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[name] = (p.grad.to_dense() if p.grad.is_sparse else p.grad).clone()
    return logits.detach(), out


@parametrize('variant', ['plain', 'mha', 'shared', 'narrow', 'everything'])
def test_model_matches_oracle(variant):
    overrides = {
        'plain': {},
        'mha': dict(new_MHA=True),
        'shared': dict(weight_sharing=2),
        'narrow': dict(hard_attn_n_c=2, n_c=4),
        'everything': dict(new_MHA=True, weight_sharing=2, hard_attn_n_c=2, n_c=4,
                           layernorm=True, comparisons_per_bit=2, tree_weights=True),
    }[variant]
    model, cfg = make_model(**overrides)
    rng = np.random.default_rng(4)
    tokens = torch.tensor(rng.integers(0, cfg.vocab_size_input, (3, cfg.context_size)))
    targets = torch.tensor(rng.integers(0, cfg.vocab_size_output, (3, cfg.context_size)))

    got_logits, got = _model_grads(model, cfg, tokens, targets)
    original = modules.lut_lookup
    modules.lut_lookup = _reference_lookup
    try:
        want_logits, want = _model_grads(model, cfg, tokens, targets)
    finally:
        modules.lut_lookup = original

    torch.testing.assert_close(got_logits, want_logits)
    assert set(got) == set(want)
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=1e-4, atol=1e-5,
                                   msg=name)
    pe_grads = [g for n, g in got.items() if n.endswith('positional_encoding')]
    assert pe_grads and any((g != 0).any() for g in pe_grads)


def test_extension_shapes():
    model, cfg = make_model(new_MHA=True, weight_sharing=2, hard_attn_n_c=2, n_c=4)
    head = model.layers[0].heads[0]
    assert isinstance(head, HardAttentionHead)
    assert head.V.n_c == 2 and head.V.total_n_c == 2 * 2 + cfg.positional_dim
    assert tuple(head.V.S.shape) == (2, 1 << (2 * 2 + 3), cfg.embedding_dim // 2)
    assert tuple(head.V.a_arr.shape) == (cfg.n_t, 2)
    assert tuple(head.positional_encoding.shape) == (cfg.context_size, cfg.n_t, 3)
    assert model.layers[0].mha_o is not None


def test_position_zero_gets_no_attention():
    model, cfg = make_model(num_layers=1)
    head = model.layers[0].heads[0]
    x = torch.randn(2, cfg.context_size, cfg.embedding_dim)
    out = head(x, cfg.gradient_pair_selection)
    assert (out[:, 0] == 0).all() and (out[:, 1:] != 0).any()


def test_frozen_pe_is_shared_across_trees_and_not_trained():
    model, cfg = make_model(freeze_PE=True, PE_init_scale=0.5)
    pe = model.layers[0].heads[0].positional_encoding
    assert not pe.requires_grad
    assert (pe == pe[:, :1]).all()


# ---------------------------------------------------------------------------
# Configuration and trainer wiring
# ---------------------------------------------------------------------------

@parametrize('bad', [dict(PE_only=True), dict(qkpe_direct_sign=True),
                     dict(normalize_qk=True), dict(stack_heads=True),
                     dict(gradient_pair_selection='cosine-distance'),
                     dict(hard_attn_n_c=0), dict(n_c=11)])
def test_invalid_hard_configs_are_refused(bad):
    kwargs = dict(BASE)
    kwargs.update(bad)
    try:
        LutConfig(**kwargs)
    except ConfigError:
        return
    raise AssertionError(f'{bad} should be refused with attention=hard')


def test_hard_attn_n_c_requires_hard():
    kwargs = dict(BASE, attention='new', hard_attn_n_c=2)
    try:
        LutConfig(**kwargs)
    except ConfigError:
        return
    raise AssertionError('hard_attn_n_c without attention=hard should be refused')


def test_parser_reaches_config():
    args = build_parser().parse_args(['dummy.txt', '--attention', 'hard',
                                      '--hard-attn-n-c', '2', '--n-c', '5'])
    args.vocab_size_input, args.vocab_size_output, args.stack_heads = 23, 7, False
    cfg = LutConfig.from_namespace(args)
    assert cfg.attention == 'hard' and cfg.hard_attn_bits == 2
    assert cfg.to_dict()['attention'] == 'hard'
    default = build_parser().parse_args(['dummy.txt'])
    assert default.attention == 'new' and default.hard_attn_n_c is None


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
