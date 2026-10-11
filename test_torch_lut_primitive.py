"""Validate lut_torch's LUT primitive against main.py's NumPy/Numba kernels.

These tests are the foundation of the port: everything in lut_torch/modules.py is built
out of `lut_lookup`, so if this file is green the remaining risk is wiring, not maths.

Run with:  python -m pytest -q test_torch_lut_primitive.py
"""

import random

import numpy as np
import torch

import main
from lut_torch import comparisons
from lut_torch._testing import parametrize, run_all
from lut_torch.functional import (
    SELECTION_CLOSEST,
    SELECTION_COSINE,
    LutGroup,
    lut_lookup,
    lut_lookup_reference,
)


class Args:
    """Minimal stand-in for main.py's argparse namespace."""

    def __init__(self, **kw):
        self.lut_comparison_mode = 'random'
        self.comparisons_per_bit = 1
        self.fp16 = False
        self.gradient_pair_selection = 'closest'
        self.__dict__.update(kw)


def seeded(seed=1234):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Construction parity: same RNG stream as main.py
# ---------------------------------------------------------------------------

@parametrize('n_t,raw_n_c,input_dim', [(4, 3, 16), (7, 6, 32), (2, 12, 8)])
def test_comparison_arrays_match_main(n_t, raw_n_c, input_dim):
    seeded()
    a_ref, b_ref = main.build_comparison_arrays('random', n_t, raw_n_c, input_dim)
    seeded()
    a_new, b_new = comparisons.build_random_comparison_arrays(n_t, raw_n_c, input_dim)
    np.testing.assert_array_equal(a_ref, a_new)
    np.testing.assert_array_equal(b_ref, b_new)


@parametrize('shape,pd,scale', [((8, 5), 5, 1.0), ((16, 8), 3, 0.25)])
def test_pe_init_matches_main(shape, pd, scale):
    args = Args(PE_init_scale=scale, positional_dim=pd)
    seeded()
    ref = main.initialize_positional_encoding(shape, args)
    seeded()
    new = comparisons.initialize_positional_encoding(shape, scale, pd)
    np.testing.assert_array_equal(ref, new)


def test_random_vector_matches_main():
    seeded()
    ref = main.random_vector(37, 1.0)
    seeded()
    new = comparisons.random_vector(37, 1.0)
    np.testing.assert_array_equal(ref, new)


def test_causal_pair_indices_match_main_layout():
    cs = 6
    tok_i, tok_j = comparisons.causal_pair_indices(cs)
    expected = [(i, j) for i in range(1, cs) for j in range(i)]
    assert list(zip(tok_i.tolist(), tok_j.tolist())) == expected
    for pair, (i, j) in enumerate(expected):
        assert pair == i * (i - 1) // 2 + j


# ---------------------------------------------------------------------------
# Simple LUT: forward, input gradient and table gradient vs main.py
# ---------------------------------------------------------------------------

def build_reference_lut(n_t, n_c, y_dim, input_dim, fan_in, share=1, seed=7):
    """Build a main.LUT plus the matching torch tensors."""
    args = Args(n_t=n_t, n_c=n_c, embedding_dim=input_dim, comparisons_per_bit=fan_in)
    lut = main.LUT(n_t)
    seeded(seed)
    trees = None
    s_n_t = None
    if share > 1:
        s_n_t = n_t // share
        trees = np.repeat(np.arange(s_n_t, dtype=np.int32), share)
    main.build_LUT_custom(lut, n_c, y_dim, args, n_t=n_t, n_c=n_c,
                          input_dim=input_dim, s_n_t=s_n_t, trees=trees)
    lut.S[:] = np.random.randn(*lut.S.shape).astype(np.float32)
    return lut, args


def torch_simple_forward(lut, x_np, grad_np, selection, use_reference=False):
    """Run the torch primitive over a single-sequence batch; return y, x.grad, S.grad."""
    n_bins = lut.S.shape[1]
    y_dim = lut.S.shape[2]
    S_flat = torch.tensor(lut.S.reshape(-1, y_dim), dtype=torch.float32,
                          requires_grad=True)
    trees = torch.tensor(lut.trees, dtype=torch.int64)
    a = torch.tensor(lut.a_arr, dtype=torch.int64)
    b = torch.tensor(lut.b_arr, dtype=torch.int64)

    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    u = (x[:, a] - x[:, b]).unsqueeze(0)          # (1, CS, N_T, raw)
    n_c = lut.bitmasks.shape[0]
    group = LutGroup(n_bits=n_c, fan_in=lut.comparisons_per_bit, bit_offset=0)

    capture = {}
    if use_reference:
        y = lut_lookup_reference(S_flat, trees, n_bins, [group], [u], capture=capture)
    else:
        y = lut_lookup(S_flat, trees, n_bins, [group], [u], selection=selection,
                       capture=capture)

    grad_y = torch.tensor(grad_np, dtype=torch.float32).unsqueeze(0)
    (y * grad_y).sum().backward()
    s_grad = S_flat.grad
    if s_grad.is_sparse:
        s_grad = s_grad.to_dense()
    return y.detach()[0].numpy(), x.grad.numpy(), s_grad.numpy(), capture


@parametrize('selection', [SELECTION_CLOSEST, SELECTION_COSINE])
@parametrize('fan_in', [1, 2, 3])
@parametrize('share', [1, 2])
def test_simple_lut_matches_main(selection, fan_in, share):
    cs, n_t, n_c, y_dim, ed = 5, 4, 3, 6, 12
    lut, args = build_reference_lut(n_t, n_c, y_dim, ed, fan_in, share=share)
    args.context_size = cs
    args.gradient_pair_selection = selection

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((cs, ed)).astype(np.float32)
    grad_np = rng.standard_normal((cs, y_dim)).astype(np.float32)

    # --- main.py reference ---
    j = np.zeros((cs, n_t), dtype=np.int32)
    r_min = np.zeros((cs, n_t), dtype=np.int32)
    u_min = np.zeros((cs, n_t), dtype=np.float32)
    main.cache_index_batch(lut, x_np, j, r_min, u_min)
    y_ref = lut.S[lut.trees, j].sum(axis=1)

    x_grad_ref = np.zeros((cs, ed), dtype=np.float32)
    s_before = lut.S.copy()
    main.learning_rate = 1.0
    main._lut_backward_to_input(lut, j, r_min, u_min, grad_np.copy(), x_grad_ref, args,
                                skip_s_update=False, input_values=x_np)
    s_grad_ref = (s_before - lut.S).reshape(-1, y_dim)
    lut.S[:] = s_before

    # --- torch ---
    y_new, x_grad_new, s_grad_new, capture = torch_simple_forward(
        lut, x_np, grad_np, selection)

    np.testing.assert_array_equal(capture['j'][0].numpy(), j)
    np.testing.assert_allclose(y_new, y_ref, rtol=0, atol=1e-5)
    np.testing.assert_allclose(x_grad_new, x_grad_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(s_grad_new, s_grad_ref, rtol=1e-4, atol=1e-5)


@parametrize('fan_in', [1, 2])
def test_reference_matches_function_closest(fan_in):
    """The pure-autograd oracle must agree with the custom Function."""
    cs, n_t, n_c, y_dim, ed = 5, 4, 3, 6, 12
    lut, _ = build_reference_lut(n_t, n_c, y_dim, ed, fan_in)
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((cs, ed)).astype(np.float32)
    grad_np = rng.standard_normal((cs, y_dim)).astype(np.float32)

    y_a, xg_a, sg_a, cap_a = torch_simple_forward(lut, x_np, grad_np, SELECTION_CLOSEST)
    y_b, xg_b, sg_b, cap_b = torch_simple_forward(lut, x_np, grad_np, SELECTION_CLOSEST,
                                                  use_reference=True)
    np.testing.assert_array_equal(cap_a['j'].numpy(), cap_b['j'].numpy())
    np.testing.assert_allclose(y_a, y_b, rtol=0, atol=1e-6)
    np.testing.assert_allclose(xg_a, xg_b, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sg_a, sg_b, rtol=1e-5, atol=1e-6)


def test_reference_surrogate_is_zero_in_forward():
    """The phi trick must not perturb the forward value."""
    cs, n_t, n_c, y_dim, ed = 4, 3, 3, 5, 10
    lut, _ = build_reference_lut(n_t, n_c, y_dim, ed, 1)
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((cs, ed)).astype(np.float32)
    j = np.zeros((cs, n_t), dtype=np.int32)
    r_min = np.zeros((cs, n_t), dtype=np.int32)
    u_min = np.zeros((cs, n_t), dtype=np.float32)
    main.cache_index_batch(lut, x_np, j, r_min, u_min)
    y_ref = lut.S[lut.trees, j].sum(axis=1)

    y_new, _, _, _ = torch_simple_forward(
        lut, x_np, np.zeros((cs, y_dim), dtype=np.float32), SELECTION_CLOSEST,
        use_reference=True)
    np.testing.assert_allclose(y_new, y_ref, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Batching: a batch of B sequences equals B independent single-sequence runs
# ---------------------------------------------------------------------------

def test_batch_equals_sum_of_singles():
    cs, n_t, n_c, y_dim, ed, batch = 5, 4, 3, 6, 12, 3
    lut, _ = build_reference_lut(n_t, n_c, y_dim, ed, 2)
    rng = np.random.default_rng(5)
    xs = rng.standard_normal((batch, cs, ed)).astype(np.float32)
    gs = rng.standard_normal((batch, cs, y_dim)).astype(np.float32)

    n_bins = lut.S.shape[1]
    trees = torch.tensor(lut.trees, dtype=torch.int64)
    a = torch.tensor(lut.a_arr, dtype=torch.int64)
    b = torch.tensor(lut.b_arr, dtype=torch.int64)
    group = LutGroup(n_bits=n_c, fan_in=lut.comparisons_per_bit, bit_offset=0)

    S_flat = torch.tensor(lut.S.reshape(-1, y_dim), requires_grad=True)
    x = torch.tensor(xs, requires_grad=True)
    u = x[:, :, a] - x[:, :, b]
    y = lut_lookup(S_flat, trees, n_bins, [group], [u])
    (y * torch.tensor(gs)).sum().backward()
    batched_x_grad = x.grad.numpy()
    batched_s_grad = S_flat.grad.to_dense().numpy()

    single_s_grad = np.zeros_like(batched_s_grad)
    for i in range(batch):
        S_i = torch.tensor(lut.S.reshape(-1, y_dim), requires_grad=True)
        x_i = torch.tensor(xs[i:i + 1], requires_grad=True)
        u_i = x_i[:, :, a] - x_i[:, :, b]
        y_i = lut_lookup(S_i, trees, n_bins, [group], [u_i])
        (y_i * torch.tensor(gs[i:i + 1])).sum().backward()
        np.testing.assert_allclose(x_i.grad.numpy()[0], batched_x_grad[i],
                                   rtol=1e-5, atol=1e-6)
        single_s_grad += S_i.grad.to_dense().numpy()
    np.testing.assert_allclose(single_s_grad, batched_s_grad, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Tree weights: y = sum_t w[t] * S[trees[t], j[t]]
# ---------------------------------------------------------------------------

def weighted_case(seed=0, share=1, fan_in=1, stacked=False, batch=2, cs=5, n_t=4,
                  n_c=3, y_dim=6):
    """Random inputs for a weighted lookup. ``stacked`` gives each batch row its own
    table slab via a ``(batch, N_T)`` tree map, with ``(batch, N_T)`` weights to match."""
    g = torch.Generator().manual_seed(seed)
    s_n_t = n_t // share
    n_bins = 2 ** n_c
    trees = torch.arange(n_t) // share
    if stacked:
        trees = trees + s_n_t * torch.arange(batch).unsqueeze(1)
    n_slabs = batch if stacked else 1
    return dict(
        S=torch.randn(n_slabs * s_n_t * n_bins, y_dim, generator=g),
        trees=trees,
        n_bins=n_bins,
        group=LutGroup(n_bits=n_c, fan_in=fan_in, bit_offset=0),
        u=torch.randn(batch, cs, n_t, n_c * fan_in, generator=g),
        w=torch.randn(*((batch, n_t) if stacked else (n_t,)), generator=g),
        grad_y=torch.randn(batch, cs, y_dim, generator=g),
    )


def run_weighted(case, weights='learned', reference=False, **kw):
    """Forward + backward; returns y and every gradient, densified.

    ``weights``: 'learned' (w requires grad), 'none' (unweighted call), or a tensor to use
    in place of ``case['w']``.
    """
    S = case['S'].clone().requires_grad_(True)
    u = case['u'].clone().requires_grad_(True)
    w = None
    if isinstance(weights, torch.Tensor):
        w = weights.clone().requires_grad_(True)
    elif weights == 'learned':
        w = case['w'].clone().requires_grad_(True)
    fn = lut_lookup_reference if reference else lut_lookup
    y = fn(S, case['trees'], case['n_bins'], [case['group']], [u], tree_weights=w, **kw)
    (y * case['grad_y']).sum().backward()
    S_grad = S.grad.to_dense() if S.grad.is_sparse else S.grad
    return dict(y=y.detach(), S=S_grad, u=u.grad,
                w=None if w is None else w.grad)


def assert_close_dicts(a, b, rtol=1e-5, atol=1e-6):
    assert set(a) == set(b)
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
            continue
        torch.testing.assert_close(a[key], b[key], rtol=rtol, atol=atol, msg=key)


@parametrize('fan_in', [1, 2])
@parametrize('share', [1, 2])
@parametrize('stacked', [False, True])
@parametrize('table_grad', ['dense', 'sparse'])
def test_weighted_matches_reference_closest(fan_in, share, stacked, table_grad):
    """Forward and all three gradients (S, margins, w) against the autograd oracle.

    Runs under the default kernel backend, so on CPU float32 it also checks that a
    weighted call does not reach the (unweighted) Numba kernels.
    """
    case = weighted_case(seed=1, share=share, fan_in=fan_in, stacked=stacked)
    assert_close_dicts(run_weighted(case, table_grad=table_grad),
                       run_weighted(case, reference=True))


def test_weighted_forward_is_the_weighted_sum():
    case = weighted_case(seed=2)
    capture = {}
    y = lut_lookup(case['S'], case['trees'], case['n_bins'], [case['group']],
                   [case['u']], capture=capture, tree_weights=case['w'])
    rows = case['S'][case['trees'] * case['n_bins'] + capture['j']]   # (B, P, N_T, y)
    torch.testing.assert_close(y, (rows * case['w'][:, None]).sum(-2))


@parametrize('selection', [SELECTION_CLOSEST, SELECTION_COSINE])
def test_unit_weights_reproduce_unweighted(selection):
    case = weighted_case(seed=3, fan_in=2)
    ones = torch.ones_like(case['w'])
    weighted = run_weighted(case, weights=ones, selection=selection)
    plain = run_weighted(case, weights='none', selection=selection)
    weighted.pop('w')
    plain.pop('w')
    assert_close_dicts(weighted, plain, rtol=0, atol=1e-6)


@parametrize('selection', [SELECTION_CLOSEST, SELECTION_COSINE])
def test_weights_scale_each_tree_separately(selection):
    """With one physical tree per slot and per-tree margins, tree t's table rows and
    margin gradients are exactly w[t] times the unweighted ones -- including under
    cosine selection, whose choice of comparison must not move."""
    case = weighted_case(seed=4, fan_in=2)
    w = case['w']
    weighted = run_weighted(case, selection=selection)
    plain = run_weighted(case, weights='none', selection=selection)

    torch.testing.assert_close(weighted['u'], plain['u'] * w[:, None],
                               rtol=1e-5, atol=1e-6)
    n_bins, y_dim = case['n_bins'], case['S'].shape[-1]
    torch.testing.assert_close(weighted['S'].view(-1, n_bins, y_dim),
                               plain['S'].view(-1, n_bins, y_dim) * w[:, None, None],
                               rtol=1e-5, atol=1e-6)


@parametrize('stacked', [False, True])
def test_weighted_chunked_backward_matches_unchunked(stacked):
    from lut_torch import functional as fn
    case = weighted_case(seed=5, fan_in=2, stacked=stacked)
    full = run_weighted(case, table_grad='dense')
    saved = fn.GATHER_BUDGET_ELEMS
    fn.GATHER_BUDGET_ELEMS = 1
    try:
        chunked = run_weighted(case, table_grad='dense')
    finally:
        fn.GATHER_BUDGET_ELEMS = saved
    assert_close_dicts(full, chunked)


def test_weight_grad_without_margin_grad():
    """grad_w must not depend on the margin path having computed S[j] . g."""
    case = weighted_case(seed=6)
    w = case['w'].clone().requires_grad_(True)
    y = lut_lookup(case['S'], case['trees'], case['n_bins'], [case['group']],
                   [case['u']], tree_weights=w)
    (y * case['grad_y']).sum().backward()
    torch.testing.assert_close(w.grad, run_weighted(case, reference=True)['w'])


def test_tree_weights_shape_is_checked():
    case = weighted_case(seed=7)
    for bad in (torch.ones(case['w'].shape[0] + 1), torch.ones(2, 3, case['w'].shape[0]),
                torch.ones(case['w'].shape, dtype=torch.float64)):
        try:
            lut_lookup(case['S'], case['trees'], case['n_bins'], [case['group']],
                       [case['u']], tree_weights=bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted tree_weights {tuple(bad.shape)} {bad.dtype}')


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
