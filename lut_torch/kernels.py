"""Fused Numba kernels for the ``closest``-selection LUT path.

``LUTFunction`` (see :mod:`lut_torch.functional`) is correct but, at batch 1, dominated by
framework overhead: a small LUT call (Q/K/V/FFN/output) does on the order of 10^4 float
ops of real work behind ~300 ATen dispatches, and the Final table's backward materialises
a ``(B, n_pairs, N_T, y_dim)`` gather of ``S[j]`` purely to contract it down to
``(B, n_pairs, N_T)``. Neither is fixable by rearranging torch ops. This module reproduces
the same forward/backward as fused, race-free, allocation-free Numba kernels and is a drop-
in accelerator for ``LUTFunction`` when the call qualifies (see :func:`supported`).

Every invariant here must match ``functional.py`` and, transitively, ``main.py``'s kernels
exactly -- see ``lut_torch/AGENTS.md``, section "Kernels", for the full list and why each
one matters. The two easy to get silently wrong:

* the row bit is ``u > 0`` (strict; ``u == 0`` gives bit 0) and the argmin tie-break keeps
  the FIRST minimum (replace only on strict ``<``);
* the surrogate factor at ``u == 0`` is ``+0.5`` (``sign(0) = -1`` in ``main.py``).

``fastmath`` is deliberately never used: it lets the compiler reorder float reductions,
which risks the bit-exact parity these kernels exist to preserve, and it was measured to
cost as much as it saves (see the perf-history entry in AGENTS.md).

Only ``closest`` selection is implemented. ``cosine-distance`` scores candidates with the
upstream gradient and stays on the torch path in ``functional.py`` unconditionally.
"""

import contextlib

import numpy as np
import torch

try:
    from numba import njit, prange
    _NUMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only without numba installed
    njit = None
    prange = range
    _NUMBA_IMPORT_ERROR = exc

KERNELS_AUTO = 'auto'
KERNELS_NUMBA = 'numba'
KERNELS_TORCH = 'torch'
KERNEL_BACKEND_MODES = (KERNELS_AUTO, KERNELS_NUMBA, KERNELS_TORCH)

_backend = KERNELS_AUTO


def set_kernel_backend(mode):
    """Select the LUT execution backend: 'auto' (default), 'numba', or 'torch'.

    A module-level switch rather than a ``lut_lookup`` argument on purpose: it is a pure
    performance choice with identical numerics, and adding an argument to
    ``LUTFunction.apply`` would shift the ``needs_input_grad`` offset that ``backward``
    indexes from (margins start at 8 today; see ``functional._MARGINS_ARG0``) -- a
    change that fails silently as a wrong gradient, not a crash.
    """
    global _backend
    if mode not in KERNEL_BACKEND_MODES:
        raise ValueError(f'kernel backend must be one of {KERNEL_BACKEND_MODES}')
    if mode == KERNELS_NUMBA and njit is None:
        raise RuntimeError(f'numba is not importable: {_NUMBA_IMPORT_ERROR}')
    _backend = mode


def kernel_backend():
    return _backend


def numba_available():
    return njit is not None


# Numba's parallel dispatch has a real per-call cost -- waking/joining the tbb thread
# pool -- that a small kernel cannot amortise. Measured on the production shapes: the
# small LUTs (Q/K/V/FFN/output, ~1024 visits) run FASTER capped to 2-8 threads than at 16
# (620 us vs 805 us), while the Final table (~16000 visits) genuinely wants all 16 (2374
# us torch vs 1305 us at 16 threads). So thread count is chosen per call from the work
# size, not fixed at the configured maximum. `numba.set_num_threads` is cheap to toggle
# (~2.5 us measured) so doing this on every call is not itself a cost worth avoiding.
_ITEMS_PER_THREAD = 300
_configured_threads = None


def set_kernel_threads(n):
    """Ceiling for the adaptive per-call thread count -- call once from the trainer,
    the same value passed to ``torch.set_num_threads`` / ``numba.set_num_threads``."""
    global _configured_threads
    _configured_threads = max(1, n)


def _threads_for(n_items):
    ceiling = _configured_threads or 1
    return max(1, min(ceiling, n_items // _ITEMS_PER_THREAD))


def _layout(groups, margins):
    """Classify a call into one of the two shapes the kernels understand, or None.

    Layout 1: exactly one group, no ``pair_index``.
    Layout 2: exactly two groups; group 0 has no ``pair_index`` (tree-aligned or
    tree-broadcast), group 1 has a ``pair_index`` (the Final table's V side). The reverse
    combination (paired + broadcast) does not occur in the model and is not handled.
    """
    if len(groups) == 1 and groups[0].pair_index is None:
        return 1
    if (len(groups) == 2
            and groups[0].pair_index is None
            and groups[1].pair_index is not None
            and groups[1].pair_ptr is not None
            and groups[1].pair_list is not None):
        return 2
    return None


def supported(S_flat, trees, groups, margins, selection):
    """Whether this call can run on the Numba kernels instead of the torch path."""
    if _backend == KERNELS_TORCH:
        return False
    if _backend == KERNELS_AUTO and njit is None:
        return False
    if _backend == KERNELS_NUMBA and njit is None:
        raise RuntimeError(f'numba kernel backend requested but not importable: '
                           f'{_NUMBA_IMPORT_ERROR}')
    from .functional import SELECTION_CLOSEST
    if selection != SELECTION_CLOSEST:
        return False
    if S_flat.device.type != 'cpu' or S_flat.dtype != torch.float32:
        return False
    if trees.device.type != 'cpu':
        return False
    for m in margins:
        if m.device.type != 'cpu' or m.dtype != torch.float32:
            return False
    return _layout(groups, margins) is not None


def _np(t):
    """Contiguous float32/int64 numpy view, zero-copy when already contiguous."""
    return t.contiguous().numpy()


def _tree_rows_np(trees, batch):
    """``trees`` as a ``(batch, N_T)`` int64 array, whatever form the caller passed.

    The kernels index ``trees[b, t]`` unconditionally so that one call can cover several
    stacked tables (a layer's 8 attention heads folded into the batch axis; see
    :func:`lut_torch.functional._tree_rows`). A plain ``(N_T,)`` map is broadcast to every
    row -- the array is a few hundred bytes, so materialising it per call is free next to
    branching inside the kernel on every element.
    """
    rows = trees.to(torch.int64)
    if rows.dim() == 1:
        rows = rows.unsqueeze(0).expand(batch, rows.shape[0])
    return _np(rows)


# =============================================================================
# Kernels
# =============================================================================

if njit is not None:

    @njit(parallel=True, cache=True)
    def _k_bits_argmin(u, out_j, out_r, n_bits, fan, shift):
        """Row bits (XOR-parity, shifted by ``shift``) and closest-comparison index.

        ``u``: (B, P, T, n_bits*fan) f32. Parallel over (b, p, t): each iteration owns
        exactly its own ``out_j[b,p,t]`` / ``out_r[b,p,t]`` cell, so this is race-free
        regardless of iteration order.
        """
        B, P, T, _ = u.shape
        for w in prange(B * P * T):
            b = w // (P * T)
            p = (w // T) % P
            t = w % T
            j = 0
            best = np.float32(1e38)
            rmin = 0
            for c in range(n_bits):
                par = 0
                for s in range(fan):
                    r = c * fan + s
                    val = u[b, p, t, r]
                    if val > np.float32(0.0):      # strict: u == 0 gives bit 0
                        par ^= 1
                    au = abs(val)
                    if au < best:                  # strict <: FIRST minimum wins
                        best = au
                        rmin = r
                if par:
                    j |= (1 << c)
            out_j[b, p, t] = j << shift
            out_r[b, p, t] = rmin

    @njit(parallel=True, cache=True)
    def _k_gather_sum(S, trees, j, n_bins, y_out):
        """``y_out[b,p,:] = sum_t S[trees[t]*n_bins + j[b,p,t], :]``, ascending t.

        Parallel over (b, p): each iteration owns the whole output row, so the tree sum
        stays in ascending order (matching ``embedding_bag``'s float32 rounding) with no
        cross-iteration writes.
        """
        B, P, T = j.shape
        ydim = S.shape[1]
        for w in prange(B * P):
            b = w // P
            p = w % P
            for d in range(ydim):
                y_out[b, p, d] = np.float32(0.0)
            for t in range(T):
                row = trees[b, t] * n_bins + j[b, p, t]
                for d in range(ydim):
                    y_out[b, p, d] += S[row, d]

    @njit(parallel=True, cache=True)
    def _k_combine2(j0, r0_bcast, j1, pair_index, j_out):
        """Combine layout-2 groups: ``j_out[b,p,t] = j0[b, p or 0, t] | j1[b, src, t]``.

        ``r0_bcast`` is 1 when group 0's margin has a size-1 tree axis (tree-broadcast,
        e.g. ``--qkpe-direct-sign``), else 0. ``pair_index`` maps pair -> source position
        for group 1. Parallel over (b, p, t): disjoint output cells.
        """
        B, NP, T = j_out.shape
        for w in prange(B * NP * T):
            b = w // (NP * T)
            p = (w // T) % NP
            t = w % T
            src = pair_index[p]
            t0 = 0 if r0_bcast else t
            j_out[b, p, t] = j0[b, p, t0] | j1[b, src, t]

    @njit(parallel=True, cache=True)
    def _k_sjg(S, trees, j, grad_y, n_bins, sjg_out):
        """``sjg_out[b,p,t] = S[trees[t]*n_bins + j[b,p,t], :] . grad_y[b,p,:]``.

        Replaces materialising the ``(B, P, T, y_dim)`` gather of ``S[j]`` that the torch
        path needs purely to contract away. Parallel over (b, p, t): disjoint cells.
        """
        B, P, T = j.shape
        ydim = S.shape[1]
        for w in prange(B * P * T):
            b = w // (P * T)
            p = (w // T) % P
            t = w % T
            row = trees[b, t] * n_bins + j[b, p, t]
            acc = np.float32(0.0)
            for d in range(ydim):
                acc += S[row, d] * grad_y[b, p, d]
            sjg_out[b, p, t] = acc

    @njit(inline='always')
    def _surrogate(u):
        """``-0.5*sign(u)/(1+|u|)^2`` with ``sign(0) = -1`` (so ``u == 0`` gives ``+0.5``).

        Matches ``main.py`` and ``functional._surrogate_factor`` exactly.
        """
        if u > np.float32(0.0):
            return np.float32(-0.5) / ((np.float32(1.0) + u) * (np.float32(1.0) + u))
        return np.float32(0.5) / ((np.float32(1.0) - u) * (np.float32(1.0) - u))

    @njit(parallel=True, cache=True)
    def _k_grad_aligned(S, trees, j, r, u, grad_y, sjg, n_bins, shift, fan, grad_u_out):
        """Margin gradient, tree-aligned, unpaired group (layout 1; layout 2 group 0
        when the margin is not tree-broadcast).

        Parallel over (b, p, t): each iteration owns ``grad_u_out[b,p,t,:]`` outright.
        """
        B, P, T, R = u.shape
        ydim = S.shape[1]
        for w in prange(B * P * T):
            b = w // (P * T)
            p = (w // T) % P
            t = w % T
            for c in range(R):
                grad_u_out[b, p, t, c] = np.float32(0.0)
            rr = r[b, p, t]
            flip = 1 << (shift + rr // fan)
            rowb = trees[b, t] * n_bins + (j[b, p, t] ^ flip)
            gi = np.float32(0.0)
            for d in range(ydim):
                gi += S[rowb, d] * grad_y[b, p, d]
            gi -= sjg[b, p, t]
            grad_u_out[b, p, t, rr] = gi * _surrogate(u[b, p, t, rr])

    @njit(parallel=True, cache=True)
    def _k_grad_broadcast(S, trees, j, r, u, grad_y, sjg, n_bins, shift, fan, grad_u_out):
        """Margin gradient, tree-broadcast, unpaired group (layout 2 group 0 under
        ``--qkpe-direct-sign``): one margin shared by every tree, so ``gi`` sums over t.

        Parallel over (b, p): each iteration owns ``grad_u_out[b,p,0,:]`` outright.
        """
        B, P, T = j.shape
        ydim = S.shape[1]
        R = u.shape[-1]
        for w in prange(B * P):
            b = w // P
            p = w % P
            for c in range(R):
                grad_u_out[b, p, 0, c] = np.float32(0.0)
            rr = r[b, p, 0]
            flip = 1 << (shift + rr // fan)
            tot = np.float32(0.0)
            for t in range(T):
                rowb = trees[b, t] * n_bins + (j[b, p, t] ^ flip)
                gi = np.float32(0.0)
                for d in range(ydim):
                    gi += S[rowb, d] * grad_y[b, p, d]
                tot += gi - sjg[b, p, t]
            grad_u_out[b, p, 0, rr] = tot * _surrogate(u[b, p, 0, rr])

    @njit(parallel=True, cache=True)
    def _k_grad_paired(S, trees, j, r_src, u, grad_y, sjg, pair_ptr, pair_list,
                       n_bins, shift, fan, grad_u_out):
        """Margin gradient for a paired group (layout 2 group 1: Final's V side).

        ``r_src`` / ``u`` are at the margin's own SOURCE shape (B, P_src, T, R); several
        pairs can share a source position (invariant: the closest comparison is chosen
        pre-gather). Parallel over (b, SOURCE position, t) rather than over pairs, so the
        accumulation across pairs sharing a source never races. ``pair_ptr``/``pair_list``
        are the CSR inverse of the group's ``pair_index`` (built once at construction, not
        per call): pairs mapping to source ``pos`` are ``pair_list[pair_ptr[pos]:pair_ptr[pos+1]]``.
        """
        B, Psrc, T, R = u.shape
        ydim = S.shape[1]
        for w in prange(B * Psrc * T):
            b = w // (Psrc * T)
            pos = (w // T) % Psrc
            t = w % T
            for c in range(R):
                grad_u_out[b, pos, t, c] = np.float32(0.0)
            rr = r_src[b, pos, t]
            flip = 1 << (shift + rr // fan)
            base = trees[b, t] * n_bins
            tot = np.float32(0.0)
            for k in range(pair_ptr[pos], pair_ptr[pos + 1]):
                p = pair_list[k]
                rowb = base + (j[b, p, t] ^ flip)
                gi = np.float32(0.0)
                for d in range(ydim):
                    gi += S[rowb, d] * grad_y[b, p, d]
                tot += gi - sjg[b, p, t]
            grad_u_out[b, pos, t, rr] = tot * _surrogate(u[b, pos, t, rr])

    # -------------------------------------------------------------------------
    # Serial mirrors.
    #
    # Numba's parallel dispatch has real fixed cost -- even capped to one thread it was
    # measured at ~16 us against ~11 us for a plain serial loop doing the same work, and
    # at a full 16-thread cap that grew to ~95 us (see AGENTS.md, "Kernels": the small
    # LUTs, ~1000 visits, run FASTER on a genuinely serial kernel than on a parallel one
    # capped to any thread count). Rather than hand-duplicate each kernel with `range`
    # instead of `prange` -- an easy way for the two to drift apart -- generate the serial
    # variant from the parallel one's own source, so there is exactly one body to keep
    # correct. `prange` must appear literally in the source for numba's parallelizer to
    # fire, which is why this is textual rather than passing a loop-range parameter.
    import inspect as _inspect
    import re as _re

    def _make_serial(fn, serial_name):
        import textwrap
        src = textwrap.dedent(_inspect.getsource(fn))
        src = _re.sub(r'@njit\(parallel=True,\s*cache=True\)', '@njit', src)
        src = src.replace('prange(', 'range(')
        src = _re.sub(r'^def ' + fn.__name__ + r'\(',
                      f'def {serial_name}(', src, count=1, flags=_re.M)
        ns = {'njit': njit, 'np': np, '_surrogate': _surrogate}
        exec(compile(src, f'<serial:{serial_name}>', 'exec'), ns)
        return ns[serial_name]

    _k_bits_argmin_serial = _make_serial(_k_bits_argmin, '_k_bits_argmin_serial')
    _k_gather_sum_serial = _make_serial(_k_gather_sum, '_k_gather_sum_serial')
    _k_combine2_serial = _make_serial(_k_combine2, '_k_combine2_serial')
    _k_sjg_serial = _make_serial(_k_sjg, '_k_sjg_serial')
    _k_grad_aligned_serial = _make_serial(_k_grad_aligned, '_k_grad_aligned_serial')
    _k_grad_broadcast_serial = _make_serial(_k_grad_broadcast, '_k_grad_broadcast_serial')
    _k_grad_paired_serial = _make_serial(_k_grad_paired, '_k_grad_paired_serial')


# =============================================================================
# Dispatch wrappers -- torch tensors in, torch tensors out.
# =============================================================================

# Below this many (b, p, t) work items, numba's parallel dispatch overhead exceeds the
# actual work, and the serial kernel wins outright regardless of thread count -- so it is
# used unconditionally rather than paired with a thread cap. Above it, parallel with the
# adaptive cap from _threads_for (below) is faster. Measured crossover sits well below the
# small-LUT shape (~1000 items) and well above nothing seen in this model; 4096 leaves
# margin either side without being reachable by a single small-table call.
_SERIAL_BELOW_ITEMS = 4096

class _thread_scope:
    """Cap numba's thread count to fit this call's work size, then restore.

    Only entered for calls at or above ``_SERIAL_BELOW_ITEMS`` -- below that, ``_pick``
    routes to the serial kernel and this is skipped entirely, since even a one-thread cap
    on the parallel path pays dispatch overhead the serial path does not (measured ~16 us
    vs ~11 us on a trivial kernel; see AGENTS.md). Numba's own ``set_num_threads`` is what
    gets toggled (not torch's -- the two pools are independent). A no-op when
    ``set_kernel_threads`` was never called (falls back to whatever numba is already
    configured with).
    """

    def __init__(self, n_items):
        self.n_items = n_items

    def __enter__(self):
        if _configured_threads is not None:
            import numba
            numba.set_num_threads(_threads_for(self.n_items))

    def __exit__(self, *exc):
        if _configured_threads is not None:
            import numba
            numba.set_num_threads(_configured_threads)


def _pick(parallel_fn, serial_fn, n_items):
    """Serial below the crossover, parallel (thread-capped by the caller) at or above it."""
    return serial_fn if n_items < _SERIAL_BELOW_ITEMS else parallel_fn


def forward(S_flat, trees, n_bins, groups, margins):
    """Row index ``j`` (B, P, N_T), per-group closest-comparison ``r``, and ``y``.

    Mirrors ``functional._index_and_fallback`` + the embedding-bag gather exactly:
    ``r`` is computed on ungathered (source-shape) margins, ``j`` on the gathered/combined
    shape. Returns ``(y, j, fallback)`` where ``fallback`` is a list of per-group ``r`` at
    the margin's own source shape (matching ``LUTFunction.forward``'s ``capture``).
    """
    layout = _layout(groups, margins)
    S_np = _np(S_flat)
    n_trees = trees.shape[-1]
    trees_np = _tree_rows_np(trees, margins[0].shape[0])
    # Largest work size across this call's kernels -- the bits/argmin passes are at most
    # as large as the combined (B, P, N_T) shape, so it alone decides serial-vs-parallel
    # and (when parallel) the thread cap.
    n_items = margins[0].shape[0] * (margins[-1].shape[1] if layout == 2
                                     else margins[0].shape[1]) * n_trees
    scope = _thread_scope(n_items) if n_items >= _SERIAL_BELOW_ITEMS else contextlib.nullcontext()

    with scope:
        fallback = []
        j_parts = []
        for group, margin in zip(groups, margins):
            m_np = _np(margin)
            j_out = torch.empty(m_np.shape[:3], dtype=torch.int64)
            r_out = torch.empty(m_np.shape[:3], dtype=torch.int64)
            k = _pick(_k_bits_argmin, _k_bits_argmin_serial, n_items)
            k(m_np, j_out.numpy(), r_out.numpy(),
              group.n_bits, group.fan_in, group.bit_offset)
            fallback.append(r_out)
            j_parts.append(j_out)

        if layout == 1:
            j_full = j_parts[0]
            if j_full.shape[2] == 1:
                j_full = j_full.expand(
                    j_full.shape[0], j_full.shape[1], n_trees).contiguous()
        else:
            g1 = groups[1]
            B = margins[1].shape[0]
            n_pairs = g1.pair_index.shape[0]
            bcast0 = 1 if j_parts[0].shape[2] == 1 else 0
            j0_np = _np(j_parts[0])
            j1_np = _np(j_parts[1])
            pair_index_np = _np(g1.pair_index.to(torch.int64))
            j_full = torch.empty((B, n_pairs, n_trees), dtype=torch.int64)
            k = _pick(_k_combine2, _k_combine2_serial, n_items)
            k(j0_np, bcast0, j1_np, pair_index_np, j_full.numpy())

        j_np = _np(j_full)
        y_out = torch.empty((*j_full.shape[:2], S_flat.shape[-1]), dtype=torch.float32)
        k = _pick(_k_gather_sum, _k_gather_sum_serial, n_items)
        k(S_np, trees_np, j_np, n_bins, y_out.numpy())
    return y_out, j_full, fallback


def backward_closest(S_flat, trees, n_bins, groups, margins, j, fallback, grad_y,
                     needs_grad):
    """Margin gradients for ``closest`` selection. ``needs_grad``: bool per group.

    Computes the shared ``S[j].grad_y`` dot product once (``_k_sjg``), then one gradient
    kernel per group depending on its layout: aligned, tree-broadcast, or paired. Mirrors
    ``functional._closest_group_grad`` exactly, including the tree-broadcast sum and the
    paired accumulate-back-to-source.
    """
    S_np = _np(S_flat)
    trees_np = _tree_rows_np(trees, j.shape[0])
    grad_y_np = _np(grad_y)
    j_np = _np(j)
    n_items = j.shape[0] * j.shape[1] * j.shape[2]
    scope = _thread_scope(n_items) if n_items >= _SERIAL_BELOW_ITEMS else contextlib.nullcontext()

    sjg = torch.empty(j.shape, dtype=torch.float32)
    grads = [None] * len(groups)
    with scope:
        k = _pick(_k_sjg, _k_sjg_serial, n_items)
        k(S_np, trees_np, j_np, grad_y_np, n_bins, sjg.numpy())
        sjg_np = _np(sjg)

        for gi, (group, margin) in enumerate(zip(groups, margins)):
            if not needs_grad[gi]:
                continue
            m_np = _np(margin)
            r_np = _np(fallback[gi])
            grad_out = torch.empty_like(margin)

            if group.pair_index is None:
                if margin.shape[2] == 1:
                    k = _pick(_k_grad_broadcast, _k_grad_broadcast_serial, n_items)
                    k(S_np, trees_np, j_np, r_np, m_np, grad_y_np,
                      sjg_np, n_bins, group.bit_offset, group.fan_in,
                      grad_out.numpy())
                else:
                    k = _pick(_k_grad_aligned, _k_grad_aligned_serial, n_items)
                    k(S_np, trees_np, j_np, r_np, m_np, grad_y_np, sjg_np,
                      n_bins, group.bit_offset, group.fan_in,
                      grad_out.numpy())
            else:
                pair_ptr_np = _np(group.pair_ptr.to(torch.int64))
                pair_list_np = _np(group.pair_list.to(torch.int64))
                k = _pick(_k_grad_paired, _k_grad_paired_serial, n_items)
                k(S_np, trees_np, j_np, r_np, m_np, grad_y_np, sjg_np,
                  pair_ptr_np, pair_list_np, n_bins, group.bit_offset,
                  group.fan_in, grad_out.numpy())
            grads[gi] = grad_out
    return grads
