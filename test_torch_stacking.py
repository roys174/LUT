"""Stacking a layer's heads must be a pure performance change: bit-identical, not close.

``test_torch_parity.py`` proves the *unstacked* model matches ``main.py`` exactly.  This
suite proves the stacked model matches the unstacked one exactly, which is what carries
the parity guarantee across.  Exact equality is the bar on purpose -- ``assert_close``
would hide precisely the mistakes that matter here (a reduction reordered, head outputs
summed in the wrong order, a slab offset off by one table), all of which show up as
small numerical drift rather than as an obvious break.

Run with:  .venv/bin/python test_torch_stacking.py
       or: python -m pytest -q test_torch_stacking.py
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from lut_torch._testing import parametrize, run_all
from lut_torch.config import LutConfig
from lut_torch.modules import LutModel, StackedHeads
from lut_torch.stacking import stack_state_dict

BASE = dict(
    context_size=10, embedding_dim=16, positional_dim=3, num_layers=2, num_heads=4,
    vocab_size_input=37, vocab_size_output=11, n_t=8, n_c=4, attn_qk_dim=4,
    attn_qk_n_c=4, attn_qk_n_t=8, weight_sharing=4, comparisons_per_bit=1,
    POS_task=True,
)


def build_pair(seed=7, **overrides):
    """The same model built both ways, with identical tables.

    Both are constructed from the same seed so their comparison arrays match by
    construction (``StackedHeads`` builds real heads and stacks them, precisely so the RNG
    draw order is untouched); the tables are then filled and copied across.
    """
    kwargs = dict(BASE)
    kwargs.update(overrides)

    def build(stacked):
        random.seed(seed)
        np.random.seed(seed)
        cfg = LutConfig(stack_heads=stacked, **kwargs)
        return LutModel(cfg), cfg

    plain, cfg = build(False)
    stacked, _ = build(True)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed)
        for table in plain.table_parameters():
            table.copy_(torch.randn(table.shape, generator=gen) * 0.3)
        for name, value in stack_state_dict(plain.state_dict(), stacked).items():
            dict(stacked.named_parameters()).get(name, dict(stacked.named_buffers())
                                                 .get(name)).copy_(value)
    return plain, stacked, cfg


def run_step(model, cfg, tokens, targets):
    for param in model.parameters():
        param.grad = None
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size_output), targets.reshape(-1),
                           reduction='sum')
    loss.backward()
    return logits.detach(), float(loss.detach())


def table_grads(model):
    return [(p.grad.to_dense() if p.grad is not None and p.grad.is_sparse else p.grad)
            for p in model.table_parameters()]


@parametrize('new_MHA', [False, True])
@parametrize('qkpe_direct_sign', [False, True])
def test_forward_is_bit_identical(new_MHA, qkpe_direct_sign):
    plain, stacked, cfg = build_pair(new_MHA=new_MHA,
                                     qkpe_direct_sign=qkpe_direct_sign)
    plain.eval()
    stacked.eval()
    rng = np.random.default_rng(1)
    tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size_input, (3, cfg.context_size)))
    with torch.no_grad():
        a = plain(tokens)
        b = stacked(tokens)
    assert torch.equal(a, b), f'max |diff| = {(a - b).abs().max()}'


@parametrize('PE_only', [False, True])
@parametrize('qkpe_direct_sign', [False, True])
def test_forward_is_bit_identical_pe_only(PE_only, qkpe_direct_sign):
    """PE_only skips Q/K entirely, so it exercises a separate margin path in StackedHeads."""
    plain, stacked, cfg = build_pair(PE_only=PE_only,
                                     qkpe_direct_sign=qkpe_direct_sign)
    plain.eval()
    stacked.eval()
    rng = np.random.default_rng(6)
    tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size)))
    with torch.no_grad():
        assert torch.equal(plain(tokens), stacked(tokens))


@parametrize('normalize_qk', [False, True])
@parametrize('layernorm', [False, True])
def test_forward_is_bit_identical_ablations(normalize_qk, layernorm):
    plain, stacked, cfg = build_pair(normalize_qk=normalize_qk, layernorm=layernorm,
                                     qkpe_direct_sign=True, freeze_PE=True)
    plain.eval()
    stacked.eval()
    rng = np.random.default_rng(2)
    tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size)))
    with torch.no_grad():
        assert torch.equal(plain(tokens), stacked(tokens))


def _grad_comparison(selection, num_layers, **overrides):
    """Run one step both ways; return ``{name: (stacked_grad, unstacked_grad)}``."""
    plain, stacked, cfg = build_pair(gradient_pair_selection=selection,
                                     num_layers=num_layers, qkpe_direct_sign=True,
                                     table_grad='dense', **overrides)
    rng = np.random.default_rng(3)
    tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size)))
    targets = torch.from_numpy(rng.integers(0, cfg.vocab_size_output,
                                            (2, cfg.context_size)))
    logits_p, loss_p = run_step(plain, cfg, tokens, targets)
    logits_s, loss_s = run_step(stacked, cfg, tokens, targets)
    assert torch.equal(logits_p, logits_s), 'forward must be exact regardless of depth'
    assert loss_p == loss_s

    want = stack_state_dict({n: g for (n, _), g in
                             zip(_table_names(plain), table_grads(plain))}, stacked)
    got = dict(zip([n for n, _ in _table_names(stacked)], table_grads(stacked)))
    assert set(want) == set(got), f'{sorted(want)} != {sorted(got)}'
    return {name: (got[name], want[name]) for name in want}


@parametrize('selection', ['closest', 'cosine-distance'])
def test_single_layer_gradients_are_bit_identical(selection):
    """With one layer nothing reassociates, so every table gradient must match exactly.

    This is the strict form of the claim.  It isolates the stacked lookup itself from the
    one place stacking genuinely reorders a sum -- the gradient flowing back into a
    layer's input, which only exists when there is a layer below.
    """
    for name, (got, expected) in _grad_comparison(selection, 1).items():
        assert torch.equal(got, expected), \
            f'{name}: max |diff| = {(got - expected).abs().max()}'


@parametrize('selection', ['closest', 'cosine-distance'])
@parametrize('num_layers', [2, 3])
def test_deep_gradients_match_to_float32_epsilon(selection, num_layers):
    """Below the top layer, fusing H per-head gathers reassociates the input gradient.

    Bounded at a few ULP and, importantly, not growing with depth -- see StackedHeads'
    docstring.  The bound is on the *relative* error because the absolute scale of these
    gradients varies by orders of magnitude between tables.
    """
    worst, worst_name = 0.0, ''
    for name, (got, expected) in _grad_comparison(selection, num_layers).items():
        scale = float(expected.abs().max())
        if scale == 0.0:
            assert torch.equal(got, expected), name
            continue
        rel = float((got - expected).abs().max()) / scale
        if rel > worst:
            worst, worst_name = rel, name
    assert worst < 1e-5, f'{worst_name}: relative diff {worst:.3e} exceeds float32 noise'


def _table_names(model):
    """``(state_dict name, parameter)`` for each table, in ``table_parameters()`` order."""
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [(by_id[id(p)], p) for p in model.table_parameters()]


def test_pe_gradient_matches():
    """PE gradients stack the same way the parameters do (float32-epsilon, see above)."""
    plain, stacked, cfg = build_pair(qkpe_direct_sign=False, freeze_PE=False,
                                     table_grad='dense')
    rng = np.random.default_rng(4)
    tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size)))
    targets = torch.from_numpy(rng.integers(0, cfg.vocab_size_output,
                                            (2, cfg.context_size)))
    run_step(plain, cfg, tokens, targets)
    run_step(stacked, cfg, tokens, targets)
    for layer_p, layer_s in zip(plain.layers, stacked.layers):
        want = torch.stack([h.positional_encoding.grad for h in layer_p.heads])
        torch.testing.assert_close(layer_s.heads.positional_encoding.grad, want,
                                   rtol=1e-5, atol=1e-7)


def test_stacked_uses_fewer_lut_calls():
    """The whole point: one call per table role per layer, not one per head."""
    from lut_torch import functional

    counts = {}
    original = functional.LUTFunction.apply

    def counting(*args, **kwargs):
        counts['n'] = counts.get('n', 0) + 1
        return original(*args, **kwargs)

    for stacked, key in ((False, 'plain'), (True, 'stacked')):
        plain, stack, cfg = build_pair()
        model = stack if stacked else plain
        counts.clear()
        functional.LUTFunction.apply = counting
        try:
            with torch.no_grad():
                model(torch.zeros(1, cfg.context_size, dtype=torch.int64))
        finally:
            functional.LUTFunction.apply = original
        counts[key] = counts['n']
        if stacked:
            stacked_n = counts['n']
        else:
            plain_n = counts['n']

    H, L = BASE['num_heads'], BASE['num_layers']
    assert plain_n == L * (4 * H + 1) + 1, plain_n      # Q,K,V,Final per head + FFN + out
    assert stacked_n == L * (3 + 1) + 1, stacked_n      # QK, V, Final + FFN, + out
    assert isinstance(stack.layers[0].heads, StackedHeads)


def test_checkpoint_conversion_round_trip():
    """An unstacked checkpoint must load into a stacked model and reproduce it exactly."""
    plain, stacked, cfg = build_pair()
    converted = stack_state_dict(plain.state_dict(), stacked)
    stacked.load_state_dict(converted)
    rng = np.random.default_rng(5)
    tokens = torch.from_numpy(rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size)))
    plain.eval()
    stacked.eval()
    with torch.no_grad():
        assert torch.equal(plain(tokens), stacked(tokens))


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
