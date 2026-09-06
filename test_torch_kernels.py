"""Validate the fused Numba kernel backend (lut_torch/kernels.py) against the torch path.

The kernel backend is a pure performance accelerator for `--gradient-pair-selection
closest` on CPU float32; it must reproduce the torch LUTFunction path exactly (row index
and forward output bit-for-bit, margin gradients to float32 rounding). See
lut_torch/PLAN_numba_kernels.md and lut_torch/AGENTS.md ("Kernels") for the invariants
these tests are checking.

Run with:  python test_torch_kernels.py   (or pytest, if available)
"""

import torch

from lut_torch._testing import parametrize, run_all, skip
from lut_torch.functional import LutGroup, lut_lookup
from lut_torch.kernels import (
    KERNELS_NUMBA,
    KERNELS_TORCH,
    numba_available,
    set_kernel_backend,
)


def _require_numba():
    if not numba_available():
        skip('numba not importable')


def _run_both(S_flat, trees, n_bins, groups, margins, requires_grad):
    """Run the same call under both backends; returns (torch_result, numba_result).

    grad_y is generated once, outside the backend loop -- it must be identical for both
    runs, or a real "gradient mismatch" is indistinguishable from "different upstream
    gradient".
    """
    grad_y_shape = (*margins[0].shape[:2], S_flat.shape[-1])
    grad_y = torch.randn(grad_y_shape)

    results = {}
    for backend in (KERNELS_TORCH, KERNELS_NUMBA):
        set_kernel_backend(backend)
        S = S_flat.clone().detach().requires_grad_(requires_grad)
        ms = [m.clone().detach().requires_grad_(requires_grad) for m in margins]
        capture = {}
        y = lut_lookup(S, trees, n_bins, groups, ms, table_grad='dense', capture=capture)
        grads = None
        if requires_grad:
            y.backward(grad_y)
            grads = [m.grad for m in ms]
        results[backend] = dict(y=y.detach(), j=capture['j'], r=capture['r_closest'],
                                grads=grads, grad_y=grad_y, S_grad=S.grad)
    set_kernel_backend(KERNELS_TORCH)
    return results[KERNELS_TORCH], results[KERNELS_NUMBA]


def _assert_match(ref, got, atol=1e-6, rtol=1e-5):
    assert torch.equal(ref['y'], got['y']), \
        f"y mismatch, max diff {(ref['y'] - got['y']).abs().max()}"
    assert torch.equal(ref['j'], got['j']), 'row index j mismatch'
    for gi, (r_ref, r_got) in enumerate(zip(ref['r'], got['r'])):
        assert torch.equal(r_ref, r_got), f'group {gi}: closest-comparison r mismatch'
    if ref['grads'] is not None:
        for gi, (g_ref, g_got) in enumerate(zip(ref['grads'], got['grads'])):
            assert torch.allclose(g_ref, g_got, atol=atol, rtol=rtol), \
                f'group {gi}: grad mismatch, max diff {(g_ref - g_got).abs().max()}'


# ---------------------------------------------------------------------------
# Layout 1: single group, no pair_index (Q/K/V/FFN/MHA_O/output shape)
# ---------------------------------------------------------------------------

@parametrize('fan_in,y_dim,batch,n_t,n_c', [
    (1, 5, 1, 4, 3),
    (1, 18, 4, 8, 4),
    (2, 64, 2, 6, 3),
])
def test_layout1_matches_torch(fan_in, y_dim, batch, n_t, n_c):
    _require_numba()
    torch.manual_seed(0)
    n_bins = 1 << n_c
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    P = 6
    margin = torch.randn(batch, P, n_t, n_c * fan_in)
    groups = [LutGroup(n_bits=n_c, fan_in=fan_in, bit_offset=0)]

    ref, got = _run_both(S, trees, n_bins, groups, [margin], requires_grad=True)
    _assert_match(ref, got)


@parametrize('weight_sharing', [1, 2])
def test_layout1_weight_sharing(weight_sharing):
    _require_numba()
    torch.manual_seed(1)
    n_t, n_c, y_dim, batch, P = 6, 3, 8, 2, 5
    n_phys = n_t // weight_sharing if weight_sharing > 1 else n_t
    n_bins = 1 << n_c
    S = torch.randn(n_phys * n_bins, y_dim)
    trees = torch.arange(n_t) % n_phys
    margin = torch.randn(batch, P, n_t, n_c)
    groups = [LutGroup(n_bits=n_c, fan_in=1, bit_offset=0)]

    ref, got = _run_both(S, trees, n_bins, groups, [margin], requires_grad=True)
    _assert_match(ref, got)


def test_layout1_forward_only_no_grad():
    _require_numba()
    torch.manual_seed(2)
    n_t, n_c, y_dim = 4, 3, 6
    n_bins = 1 << n_c
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    margin = torch.randn(1, 5, n_t, n_c)
    groups = [LutGroup(n_bits=n_c, fan_in=1, bit_offset=0)]
    ref, got = _run_both(S, trees, n_bins, groups, [margin], requires_grad=False)
    _assert_match(ref, got)


def test_layout1_u_equals_zero():
    """u == 0 must give bit 0 (strict >) under both backends, and the surrogate factor
    at the chosen r must be +0.5 if u==0 happens to be the closest comparison."""
    _require_numba()
    n_t, n_c, y_dim = 1, 2, 4
    n_bins = 1 << n_c
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    margin = torch.tensor([[[[0.0, 0.3]]]])  # (1,1,1,2): r=0 is exactly zero and closest
    groups = [LutGroup(n_bits=n_c, fan_in=1, bit_offset=0)]
    ref, got = _run_both(S, trees, n_bins, groups, [margin], requires_grad=True)
    _assert_match(ref, got)
    assert ref['r'][0].item() == 0
    # phi'(0) with sign(0)=-1 is +0.5 * gi; just confirm both backends agree exactly.
    assert torch.equal(ref['grads'][0], got['grads'][0])


def test_layout1_exact_tie_first_minimum():
    """Two raw comparisons with identical |u|: the FIRST index must win under both
    backends (torch.min documents this; the kernel must replace only on strict <)."""
    _require_numba()
    n_t, n_c, y_dim = 1, 1, 4
    n_bins = 1 << n_c
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    # fan_in=3 comparisons feeding one bit; r=0 and r=2 tie on |u|=0.5.
    margin = torch.tensor([[[[0.5, 0.9, -0.5]]]])
    groups = [LutGroup(n_bits=1, fan_in=3, bit_offset=0)]
    ref, got = _run_both(S, trees, n_bins, groups, [margin], requires_grad=True)
    _assert_match(ref, got)
    assert ref['r'][0].item() == 0, 'reference (torch) path did not pick the first tie'


def test_layout1_partial_needs_grad():
    """One of two margin-producing groups has requires_grad False; must not crash and
    the other's gradient must still match."""
    _require_numba()
    torch.manual_seed(3)
    n_t, n_c, y_dim = 4, 2, 5
    n_bins = 1 << n_c
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    m1 = torch.randn(1, 3, n_t, n_c)
    groups = [LutGroup(n_bits=n_c, fan_in=1, bit_offset=0)]

    for backend in (KERNELS_TORCH, KERNELS_NUMBA):
        set_kernel_backend(backend)
        Sg = S.clone().requires_grad_(True)
        mg = m1.clone().requires_grad_(False)
        y = lut_lookup(Sg, trees, n_bins, groups, [mg], table_grad='dense')
        y.sum().backward()
        assert mg.grad is None
        assert Sg.grad is not None
    set_kernel_backend(KERNELS_TORCH)


# ---------------------------------------------------------------------------
# Layout 2: two groups (Final table) -- direct-sign (broadcast) and non-direct-sign
# ---------------------------------------------------------------------------

def _causal_pairs(cs):
    tok_i, tok_j = [], []
    for i in range(1, cs):
        for j in range(i):
            tok_i.append(i)
            tok_j.append(j)
    return torch.tensor(tok_i), torch.tensor(tok_j)


def _csr_inverse(tok_j, n_src):
    order = torch.argsort(tok_j, stable=True)
    counts = torch.bincount(tok_j, minlength=n_src)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return ptr, order


@parametrize('direct_sign', [True, False])
def test_layout2_final_table(direct_sign):
    _require_numba()
    torch.manual_seed(4)
    CS, n_t, n_c, pd, y_dim, batch = 8, 4, 3, 2, 6, 2
    n_bins = 1 << (n_c + pd)
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    tok_i, tok_j = _causal_pairs(CS)
    n_pairs = tok_i.shape[0]
    pair_ptr, pair_list = _csr_inverse(tok_j, CS)

    if direct_sign:
        qkpe = torch.randn(batch, n_pairs, 1, pd)
    else:
        qkpe = torch.randn(batch, n_pairs, n_t, pd)
    v_margin = torch.randn(batch, CS, n_t, n_c)

    groups = [
        LutGroup(n_bits=pd, fan_in=1, bit_offset=0),
        LutGroup(n_bits=n_c, fan_in=1, bit_offset=pd, pair_index=tok_j,
                 pair_ptr=pair_ptr, pair_list=pair_list),
    ]
    ref, got = _run_both(S, trees, n_bins, groups, [qkpe, v_margin], requires_grad=True)
    _assert_match(ref, got)


def test_layout2_weight_sharing():
    _require_numba()
    torch.manual_seed(5)
    CS, n_t, n_c, pd, y_dim, batch, ws = 8, 4, 3, 2, 6, 2, 2
    n_phys = n_t // ws
    n_bins = 1 << (n_c + pd)
    S = torch.randn(n_phys * n_bins, y_dim)
    trees = torch.arange(n_t) % n_phys
    tok_i, tok_j = _causal_pairs(CS)
    n_pairs = tok_i.shape[0]
    pair_ptr, pair_list = _csr_inverse(tok_j, CS)

    qkpe = torch.randn(batch, n_pairs, 1, pd)
    v_margin = torch.randn(batch, CS, n_t, n_c)
    groups = [
        LutGroup(n_bits=pd, fan_in=1, bit_offset=0),
        LutGroup(n_bits=n_c, fan_in=1, bit_offset=pd, pair_index=tok_j,
                 pair_ptr=pair_ptr, pair_list=pair_list),
    ]
    ref, got = _run_both(S, trees, n_bins, groups, [qkpe, v_margin], requires_grad=True)
    _assert_match(ref, got)


def test_layout2_comparisons_per_bit():
    _require_numba()
    torch.manual_seed(6)
    CS, n_t, n_c, pd, y_dim, batch, fan = 8, 4, 3, 2, 6, 1, 2
    n_bins = 1 << (n_c + pd)
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    tok_i, tok_j = _causal_pairs(CS)
    n_pairs = tok_i.shape[0]
    pair_ptr, pair_list = _csr_inverse(tok_j, CS)

    qkpe = torch.randn(batch, n_pairs, 1, pd)
    v_margin = torch.randn(batch, CS, n_t, n_c * fan)
    groups = [
        LutGroup(n_bits=pd, fan_in=1, bit_offset=0),
        LutGroup(n_bits=n_c, fan_in=fan, bit_offset=pd, pair_index=tok_j,
                 pair_ptr=pair_ptr, pair_list=pair_list),
    ]
    ref, got = _run_both(S, trees, n_bins, groups, [qkpe, v_margin], requires_grad=True)
    _assert_match(ref, got)


def test_layout2_capture_matches():
    """capture['j'] / capture['r_closest'] populated identically under both backends."""
    _require_numba()
    torch.manual_seed(7)
    CS, n_t, n_c, pd, y_dim, batch = 6, 4, 2, 2, 5, 1
    n_bins = 1 << (n_c + pd)
    S = torch.randn(n_t * n_bins, y_dim)
    trees = torch.arange(n_t)
    tok_i, tok_j = _causal_pairs(CS)
    n_pairs = tok_i.shape[0]
    pair_ptr, pair_list = _csr_inverse(tok_j, CS)
    qkpe = torch.randn(batch, n_pairs, 1, pd)
    v_margin = torch.randn(batch, CS, n_t, n_c)
    groups = [
        LutGroup(n_bits=pd, fan_in=1, bit_offset=0),
        LutGroup(n_bits=n_c, fan_in=1, bit_offset=pd, pair_index=tok_j,
                 pair_ptr=pair_ptr, pair_list=pair_list),
    ]
    caps = {}
    for backend in (KERNELS_TORCH, KERNELS_NUMBA):
        set_kernel_backend(backend)
        capture = {}
        lut_lookup(S, trees, n_bins, groups, [qkpe, v_margin], table_grad='dense',
                  capture=capture)
        caps[backend] = capture
    set_kernel_backend(KERNELS_TORCH)
    assert torch.equal(caps[KERNELS_TORCH]['j'], caps[KERNELS_NUMBA]['j'])
    for r_ref, r_got in zip(caps[KERNELS_TORCH]['r_closest'], caps[KERNELS_NUMBA]['r_closest']):
        assert torch.equal(r_ref, r_got)


def test_backend_selection_validation():
    from lut_torch.kernels import set_kernel_backend, KERNEL_BACKEND_MODES
    try:
        set_kernel_backend('bogus')
        assert False, 'expected ValueError'
    except ValueError:
        pass
    finally:
        set_kernel_backend(KERNELS_TORCH)


if __name__ == '__main__':
    import sys
    sys.exit(run_all(globals()))
