"""Exact-parity gate: lut_torch must reproduce main.py step for step.

Both models are built in-process from the same seed (no checkpoint interchange), fed the
same tokens, and compared on:

  1. construction  -- comparison arrays, tree maps, positional encodings
  2. forward       -- logits and every intermediate row index
  3. loss
  4. gradients     -- every table, PE, and LayerNorm affine
  5. one SGD step  -- resulting table values

main.py accumulates gradients by mutating S during backward, so its "gradient" is
recovered as ``(S_before - S_after) / lr``.

Run with:  .venv/bin/python test_torch_parity.py
       or: python -m pytest -q test_torch_parity.py
"""

import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import main
from lut_torch._testing import parametrize, run_all
from lut_torch.config import LutConfig
from lut_torch.modules import LutModel

IGNORE_INDEX = main.IGNORE_INDEX
LR = 0.05


# ---------------------------------------------------------------------------
# Building the two models from one configuration
# ---------------------------------------------------------------------------

def main_args_from_config(cfg):
    """Translate a LutConfig into the argparse namespace main.py expects."""
    args = SimpleNamespace(
        context_size=cfg.context_size,
        embedding_dim=cfg.embedding_dim,
        positional_dim=cfg.positional_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        vocab_size_input=cfg.vocab_size_input,
        vocab_size_output=cfg.vocab_size_output,
        n_t=cfg.n_t,
        n_c=cfg.n_c,
        comparisons_per_bit=cfg.comparisons_per_bit,
        attn_qk_dim=cfg.attn_qk_dim,
        attn_qk_n_c=cfg.attn_qk_n_c,
        attn_qk_n_t=cfg.attn_qk_n_t,
        lut_comparison_mode='random',
        fp16=False,
        new_attention=cfg.attention == 'new',
        new_MHA=cfg.new_MHA,
        mha_head_dim=cfg.mha_head_dim,
        weight_sharing=cfg.weight_sharing,
        final_n_t=cfg.final_n_t,
        final_share_m=cfg.final_share_m,
        PE_only=cfg.PE_only,
        freeze_PE=cfg.freeze_PE,
        PE_init_scale=cfg.PE_init_scale,
        qkpe_direct_sign=cfg.qkpe_direct_sign,
        normalize_qk=cfg.normalize_qk,
        qk_norm_eps=cfg.qk_norm_eps,
        qk_norm_target_std=np.float32(cfg.qk_norm_target_std),
        gradient_pair_selection=cfg.gradient_pair_selection,
        layernorm=cfg.layernorm,
        dropout=cfg.dropout,
        POS_task=cfg.POS_task,
        factored_output=False,
        soft_attention=False,
        batch_size=1,
        head_workers=0,
    )
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim
    return args


def build_pair(cfg, seed=42):
    """Build a main.Model and a LutModel from the same RNG seed."""
    args = main_args_from_config(cfg)

    random.seed(seed)
    np.random.seed(seed)
    reference = main.Model(args)
    main.build_Model(reference, args)

    random.seed(seed)
    np.random.seed(seed)
    model = LutModel(cfg)
    return reference, args, model


def table_pairs(reference, model, cfg):
    """Yield (label, main.LUT, LUTTable) for every table in the model."""
    for l in range(cfg.num_layers):
        yield f'ffn_{l}', reference.FFN[l], model.layers[l].ffn
        if cfg.new_MHA:
            yield f'mha_o_{l}', reference.MHA_O[l], model.layers[l].mha_o
        for h in range(cfg.num_heads):
            ref_head = reference.head[l][h]
            new_head = model.layers[l].heads[h]
            if cfg.attention == 'hard':
                yield f'head_{l}_{h}_V', ref_head.V, new_head.V
                continue
            if not cfg.PE_only:
                yield f'head_{l}_{h}_Q', ref_head.Q, new_head.Q
                yield f'head_{l}_{h}_K', ref_head.K, new_head.K
            yield f'head_{l}_{h}_V', ref_head.V, new_head.V
            yield f'head_{l}_{h}_Final', ref_head.Final, new_head.Final
    yield 'output', reference.output_head.unembedder, model.output_head


# ---------------------------------------------------------------------------
# Driving one training step on each side
# ---------------------------------------------------------------------------

def run_main_step(reference, args, tokens, targets, lr):
    """Forward + backward one snippet through main.py. Returns loss, logits, grads."""
    cs = args.context_size
    reference.tokens[:cs + 1] = tokens
    reference.z[:] = reference.Token_embedder[reference.tokens[:cs]]
    reference.output_head.load_targets(targets, cs)

    main.model_forward(reference, args, training=False)
    logits = reference.output_head.output.copy()
    loss = reference.output_head.training_loss(args)

    before = snapshot(reference, args)
    reference.output_head.compute_gradients(args)
    main.learning_rate = lr
    main.model_backward(reference, args)
    after = snapshot(reference, args)

    grads = {k: (before[k] - after[k]) / lr for k in before}
    return loss, logits, grads, after


def snapshot(reference, args):
    """Copy every trainable array out of a main.Model."""
    out = {}
    for l in range(args.num_layers):
        out[f'ffn_{l}'] = reference.FFN[l].S.copy()
        if args.new_MHA:
            out[f'mha_o_{l}'] = reference.MHA_O[l].S.copy()
        for h in range(args.num_heads):
            head = reference.head[l][h]
            out[f'head_{l}_{h}_V'] = head.V.S.copy()
            if args.new_attention:
                if not args.PE_only:
                    out[f'head_{l}_{h}_Q'] = head.Q.S.copy()
                    out[f'head_{l}_{h}_K'] = head.K.S.copy()
                out[f'head_{l}_{h}_Final'] = head.Final.S.copy()
            out[f'pe_{l}_{h}'] = head.Positional_encoding.copy()
        if args.layernorm:
            out[f'ln_attn_gamma_{l}'] = reference.ln_attn_gamma[l].copy()
            out[f'ln_attn_beta_{l}'] = reference.ln_attn_beta[l].copy()
            out[f'ln_ffn_gamma_{l}'] = reference.ln_ffn_gamma[l].copy()
            out[f'ln_ffn_beta_{l}'] = reference.ln_ffn_beta[l].copy()
    out['output'] = reference.output_head.unembedder.S.copy()
    return out


def run_torch_step(model, cfg, tokens, targets, lr):
    """Forward + backward the same snippet through lut_torch."""
    model.train(False)
    tok = torch.tensor(tokens[:cfg.context_size], dtype=torch.int64).unsqueeze(0)
    tgt = torch.tensor(targets[:cfg.context_size] if cfg.POS_task
                       else targets[1:cfg.context_size + 1], dtype=torch.int64)

    capture = {}
    logits = model(tok, capture=capture)
    flat = logits.reshape(-1, cfg.vocab_size_output)
    # main.py's table gradient is the SUM of per-position cross entropy (its
    # compute_gradients writes softmax - onehot with no 1/CS), while the number it
    # reports is the mean over valid positions.
    total = F.cross_entropy(flat, tgt, ignore_index=IGNORE_INDEX, reduction='sum')
    mean = F.cross_entropy(flat, tgt, ignore_index=IGNORE_INDEX, reduction='mean')

    model.zero_grad(set_to_none=True)
    total.backward()

    return float(mean.detach()), logits.detach()[0].numpy(), capture


def torch_grads(model, cfg):
    """Collect dense gradients keyed like :func:`snapshot`."""
    out = {}
    for l in range(cfg.num_layers):
        layer = model.layers[l]
        out[f'ffn_{l}'] = dense_grad(layer.ffn)
        if cfg.new_MHA:
            out[f'mha_o_{l}'] = dense_grad(layer.mha_o)
        for h in range(cfg.num_heads):
            head = layer.heads[h]
            out[f'head_{l}_{h}_V'] = dense_grad(head.V)
            if cfg.attention == 'new':
                if not cfg.PE_only:
                    out[f'head_{l}_{h}_Q'] = dense_grad(head.Q)
                    out[f'head_{l}_{h}_K'] = dense_grad(head.K)
                out[f'head_{l}_{h}_Final'] = dense_grad(head.Final)
            pe = head.positional_encoding
            out[f'pe_{l}_{h}'] = (np.zeros_like(pe.detach().numpy())
                                  if pe.grad is None else pe.grad.numpy())
        if cfg.layernorm:
            out[f'ln_attn_gamma_{l}'] = layer.ln_attn_gamma.grad.numpy()
            out[f'ln_attn_beta_{l}'] = layer.ln_attn_beta.grad.numpy()
            out[f'ln_ffn_gamma_{l}'] = layer.ln_ffn_gamma.grad.numpy()
            out[f'ln_ffn_beta_{l}'] = layer.ln_ffn_beta.grad.numpy()
    out['output'] = dense_grad(model.output_head)
    return out


def dense_grad(table):
    """Table gradient as a dense ``(trees, bins, y_dim)`` array."""
    grad = table.S_flat.grad
    if grad is None:
        return np.zeros((table.s_n_t, table.n_bins, table.y_dim), dtype=np.float32)
    if grad.is_sparse:
        grad = grad.to_dense()
    return grad.numpy().reshape(table.s_n_t, table.n_bins, table.y_dim)


# ---------------------------------------------------------------------------
# Configurations under test
# ---------------------------------------------------------------------------

BASE = dict(
    context_size=8, embedding_dim=16, positional_dim=3, num_layers=2, num_heads=2,
    vocab_size_input=23, vocab_size_output=7, n_t=4, n_c=3,
    attn_qk_dim=4, attn_qk_n_c=3, attn_qk_n_t=3, POS_task=True,
)

CONFIGS = {
    'plain': dict(),
    'mha': dict(new_MHA=True, embedding_dim=16, num_heads=2),
    'layernorm': dict(layernorm=True),
    'direct_sign': dict(qkpe_direct_sign=True),
    'normalize_qk': dict(normalize_qk=True),
    'normalize_qk_direct': dict(normalize_qk=True, qkpe_direct_sign=True),
    'freeze_pe': dict(freeze_PE=True, PE_init_scale=0.5),
    'pe_only': dict(PE_only=True),
    'pe_only_frozen': dict(PE_only=True, freeze_PE=True, PE_init_scale=0.5),
    'weight_sharing': dict(weight_sharing=2),
    'fan_in_2': dict(comparisons_per_bit=2),
    'fan_in_3_mha': dict(comparisons_per_bit=3, new_MHA=True),
    'cosine': dict(gradient_pair_selection='cosine-distance'),
    'cosine_mha_shared': dict(gradient_pair_selection='cosine-distance', new_MHA=True,
                              weight_sharing=2, comparisons_per_bit=2),
    'cosine_normalize': dict(gradient_pair_selection='cosine-distance',
                             normalize_qk=True, qkpe_direct_sign=True),
    'everything': dict(new_MHA=True, layernorm=True, qkpe_direct_sign=True,
                       normalize_qk=True, freeze_PE=True, PE_init_scale=0.5,
                       weight_sharing=2, comparisons_per_bit=2,
                       gradient_pair_selection='cosine-distance'),
    'lm_task': dict(POS_task=False),
    # main.py's original hard attention (no --new-attention). Only the structures
    # main.py has: --new-MHA, --weight-sharing and --hard-attn-n-c are torch-only
    # extensions of it and are covered by test_torch_hard_attention.py instead.
    'hard': dict(attention='hard'),
    'hard_layernorm': dict(attention='hard', layernorm=True),
    'hard_fan_in_2': dict(attention='hard', comparisons_per_bit=2),
    'hard_freeze_pe': dict(attention='hard', freeze_PE=True, PE_init_scale=0.5),
    'hard_one_head': dict(attention='hard', num_heads=1, n_t=6),
    'hard_lm_task': dict(attention='hard', POS_task=False, layernorm=True),
}


def make_config(name):
    kwargs = dict(BASE)
    kwargs.update(CONFIGS[name])
    return LutConfig(**kwargs)


def make_batch(cfg, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, cfg.vocab_size_input, size=cfg.context_size + 1)
    tokens = tokens.astype(np.int32)
    targets = rng.integers(0, cfg.vocab_size_output, size=cfg.context_size + 1)
    targets = targets.astype(np.int32)
    if cfg.POS_task:
        targets[2] = IGNORE_INDEX      # exercise the ignore-index path
    return tokens, targets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@parametrize('name', sorted(CONFIGS))
def test_construction_matches(name):
    cfg = make_config(name)
    reference, args, model = build_pair(cfg)

    for label, ref_lut, table in table_pairs(reference, model, cfg):
        np.testing.assert_array_equal(ref_lut.a_arr, table.a_arr.numpy(),
                                      err_msg=f'{label} a_arr')
        np.testing.assert_array_equal(ref_lut.b_arr, table.b_arr.numpy(),
                                      err_msg=f'{label} b_arr')
        np.testing.assert_array_equal(ref_lut.trees, table.trees.numpy(),
                                      err_msg=f'{label} trees')
        assert ref_lut.S.shape == tuple(table.S.shape), f'{label} S shape'

    np.testing.assert_array_equal(reference.Token_embedder,
                                  model.token_embedding.numpy())
    for l in range(cfg.num_layers):
        for h in range(cfg.num_heads):
            ref_head = reference.head[l][h]
            new_head = model.layers[l].heads[h]
            np.testing.assert_array_equal(
                ref_head.Positional_encoding,
                new_head.positional_encoding.detach().numpy(),
                err_msg=f'PE {l},{h}')
            if cfg.attention == 'hard':
                continue
            np.testing.assert_array_equal(ref_head._qkpe_a_arr, new_head.qkpe_a.numpy())
            np.testing.assert_array_equal(ref_head._qkpe_b_arr, new_head.qkpe_b.numpy())


@parametrize('name', sorted(CONFIGS))
def test_step_matches(name):
    cfg = make_config(name)
    reference, args, model = build_pair(cfg)

    # Random non-zero tables: zero-initialised tables would hide index errors.
    rng = np.random.default_rng(17)
    for label, ref_lut, table in table_pairs(reference, model, cfg):
        values = rng.standard_normal(ref_lut.S.shape).astype(np.float32) * 0.3
        ref_lut.S[:] = values
        with torch.no_grad():
            table.S_flat.copy_(torch.tensor(values.reshape(-1, table.y_dim)))

    tokens, targets = make_batch(cfg)

    ref_loss, ref_logits, ref_grads, ref_after = run_main_step(
        reference, args, tokens, targets, LR)
    new_loss, new_logits, _ = run_torch_step(model, cfg, tokens, targets, LR)

    np.testing.assert_allclose(new_logits, ref_logits[:cfg.context_size],
                               rtol=1e-4, atol=1e-5, err_msg=f'{name} logits')
    assert abs(new_loss - ref_loss) < 1e-5, f'{name} loss {new_loss} vs {ref_loss}'

    new_grads = torch_grads(model, cfg)

    # Guard against a vacuous pass: every table must carry a real gradient, otherwise
    # "both sides agree" would only mean "both sides are zero".
    live = [k for k, g in ref_grads.items() if np.abs(g).max() > 1e-6]
    tables = [k for k in ref_grads if not k.startswith(('pe_', 'ln_'))]
    assert set(tables) <= set(live), f'{name}: tables with no gradient: ' \
        f'{sorted(set(tables) - set(live))}'

    for key, ref_grad in sorted(ref_grads.items()):
        if key.startswith('pe_') and cfg.freeze_PE:
            continue                      # main.py skips frozen-PE updates entirely
        assert key in new_grads, f'{name}: missing gradient {key}'
        np.testing.assert_allclose(new_grads[key], ref_grad, rtol=1e-3, atol=1e-5,
                                   err_msg=f'{name} gradient {key}')

    # One SGD step must land on the same tables.
    optimizer = torch.optim.SGD([
        {'params': model.table_parameters()},
        {'params': model.dense_parameters()},
    ], lr=LR)
    optimizer.step()
    for label, _, table in table_pairs(reference, model, cfg):
        np.testing.assert_allclose(
            table.S.detach().numpy(), ref_after[label], rtol=1e-3, atol=1e-5,
            err_msg=f'{name} post-step {label}')


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
