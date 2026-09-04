"""The LUT lookup primitive: the only custom forward/backward in this package.

A LUT turns real-valued *margins* into a discrete row index and reads a vector out of a
table::

    bit_b   = XOR_k ( u[..., b*K + k] > 0 )        # K = comparisons_per_bit
    j       = sum_b bit_b << b     (per group, shifted by the group's bit_offset)
    y[p]    = sum_t S[trees[t], j[p, t], :]

Nothing about that is differentiable, so ``main.py`` defines two gradients by hand and this
module reproduces them exactly:

* **Table gradient** -- ``S[tree, j] -= lr * grad_y``, accumulated over every (position,
  tree) that landed in the bin.  That is a scatter-add.  Its representation is chosen per
  call (see ``TABLE_GRAD_*``): dense costs one row per table row, sparse costs one row per
  *visit*, and for a pair-indexed table like Final the visits outnumber the rows even at
  batch 1 -- so sparse is both bigger and slower there.

* **Input gradient** -- a surrogate.  One raw comparison ``r`` is selected per group, and
  with ``gi = (S[j_flipped] - S[j]) . grad_y``::

      grad_u[..., r] = gi * (-0.5 * sign(u) / (1 + |u|)^2)

  That factor is exactly ``dphi/du`` for ``phi(u) = 0.5 / (1 + |u|)``, which is what makes
  :func:`lut_lookup_reference` below possible.

Expressing the custom step in terms of *margins* rather than the layer input is the whole
trick: everything upstream (``u = x[a] - x[b]``, QK normalisation, PE addition,
direct-sign slicing) is then ordinary torch, and autograd derives the routing that
``main.new_attention_backward`` writes out by hand.

Why an explicit ``autograd.Function`` and not just the ``phi`` trick everywhere:
``--gradient-pair-selection cosine-distance`` scores candidate comparisons using the
*upstream gradient*, so the selection can only happen in backward.  The pure-autograd
:func:`lut_lookup_reference` therefore supports ``closest`` only and exists as a test
oracle.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

# Mirrors main.COSINE_DISTANCE_EPS.
COSINE_DISTANCE_EPS = 1e-8

SELECTION_CLOSEST = 'closest'
SELECTION_COSINE = 'cosine-distance'
SELECTION_MODES = (SELECTION_CLOSEST, SELECTION_COSINE)

# How the table gradient is represented. A LUT's gradient touches one row per
# (batch, position/pair, tree) *visit*; sparse only pays off when those visits are
# far fewer than the table's rows. For the Final table the opposite holds -- it is
# indexed per causal pair per tree, so visits exceed rows even at batch 1, and a
# sparse gradient is both larger and (measurably) several times slower to apply.
TABLE_GRAD_AUTO = 'auto'
TABLE_GRAD_DENSE = 'dense'
TABLE_GRAD_SPARSE = 'sparse'
TABLE_GRAD_MODES = (TABLE_GRAD_AUTO, TABLE_GRAD_DENSE, TABLE_GRAD_SPARSE)


class LutGroup(NamedTuple):
    """One contiguous run of row bits fed by a single margin source.

    A simple LUT (FFN, Q, K, V, MHA_O, output head) has exactly one group.  The Final
    table of new attention has two: the QKPE bits at ``bit_offset=0`` and the V-side bits
    at ``bit_offset=positional_dim``.

    Attributes:
        n_bits: learned bits contributed by this group.
        fan_in: ``--comparisons-per-bit``; raw comparison ``r`` feeds bit ``r // fan_in``.
        bit_offset: position of this group's bits within the combined row index.
        pair_index: optional ``(P,)`` gather map applied to dim 1 of the margins.  Lets a
            per-position margin tensor feed a per-pair table without the caller
            materialising the expanded ``(B, n_pairs, N_T, R)`` tensor.
    """

    n_bits: int
    fan_in: int
    bit_offset: int
    pair_index: Optional[torch.Tensor] = None

    @property
    def raw_count(self) -> int:
        return self.n_bits * self.fan_in


def _gather_pairs(margins: torch.Tensor, group: LutGroup) -> torch.Tensor:
    if group.pair_index is None:
        return margins
    return margins.index_select(1, group.pair_index)


def _first_argmin(values: torch.Tensor) -> torch.Tensor:
    """Index of the first minimum along the last dim.

    ``main.py``'s kernels scan raw comparisons in order and replace on a strict ``<``, so
    the *first* minimum wins (its NumPy fallback uses ``argmin``, which documents the same).
    ``torch.min(dim=...)`` documents first-minimum tie-breaking too, and is far faster than
    an explicit mask-and-reduce.
    """
    return values.min(dim=-1).indices


def _row_bits(gathered: torch.Tensor, group: LutGroup) -> torch.Tensor:
    """XOR-parity row index contributed by one group. ``gathered``: (..., R)."""
    positive = (gathered > 0).to(torch.int64)
    if group.fan_in > 1:
        positive = positive.reshape(*positive.shape[:-1], group.n_bits, group.fan_in)
        parity = positive.sum(dim=-1) & 1
    else:
        parity = positive
    weights = torch.arange(group.n_bits, device=gathered.device)
    return (parity << weights).sum(dim=-1)


def _index_and_fallback(margins, groups, n_trees):
    """Row index ``j`` (B, P, n_trees) plus each group's closest-margin comparison.

    Both are computed on the *ungathered* margins and gathered afterwards.  For the Final
    table that matters a lot: its V-side margins are per position (B, CS, N_T, R) while the
    table is indexed per causal pair (B, CS*(CS-1)/2, N_T, R), so doing the work before the
    gather is ~CS/2 times cheaper.
    """
    j = None
    fallback = []
    for margin, group in zip(margins, groups):
        bits = _row_bits(margin, group) << group.bit_offset
        # Returned ungathered (source shape): the closest-margin backward assembles the
        # whole gradient at source level, and only the cosine path needs it per pair.
        fallback.append(_first_argmin(margin.abs()))
        if group.pair_index is not None:
            bits = bits.index_select(1, group.pair_index)
        j = bits if j is None else (j | bits)
    j = j.expand(j.shape[0], j.shape[1], n_trees)
    return j, fallback


def _flip_mask(r: torch.Tensor, group: LutGroup) -> torch.Tensor:
    """``1 << (bit_offset + r // fan_in)`` for the selected raw comparison."""
    bit = r if group.fan_in == 1 else r.div(group.fan_in, rounding_mode='floor')
    return torch.ones_like(r) << (bit + group.bit_offset)


def _surrogate_factor(u: torch.Tensor) -> torch.Tensor:
    """``-0.5 * sign(u) / (1 + |u|)^2`` with ``sign(0) = -1``, as in ``main.py``."""
    sign = torch.where(u > 0, 1.0, -1.0).to(u.dtype)
    abs_u = torch.where(u > 0, u, -u)
    return -0.5 * sign / (1.0 + abs_u) ** 2


def _reduce_to(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Sum ``grad`` back onto ``shape``, undoing broadcasting over the tree dim."""
    if grad.shape == shape:
        return grad
    for dim, size in enumerate(shape):
        if size == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad


class LUTFunction(torch.autograd.Function):
    """Forward/backward for ``y[p] = sum_t S[trees[t], j[p, t]]``."""

    @staticmethod
    def forward(ctx, S_flat, trees, n_bins, groups, selection, table_grad, capture,
                *margins):
        n_trees = trees.shape[0]
        with torch.no_grad():
            j, fallback = _index_and_fallback(margins, groups, n_trees)
            idx = trees.to(torch.int64) * n_bins + j
            # embedding_bag sums the tree axis without materialising the
            # (B, P, N_T, y_dim) gather that embedding(...).sum(-2) would.
            y = F.embedding_bag(idx.reshape(-1, n_trees), S_flat, mode='sum').view(
                *idx.shape[:-1], S_flat.shape[-1])

        ctx.save_for_backward(S_flat, trees, j, *fallback, *margins)
        ctx.groups = groups
        ctx.n_bins = n_bins
        ctx.selection = selection
        ctx.table_grad = table_grad

        if capture is not None:
            capture['j'] = j
            capture['r_closest'] = fallback
        return y

    @staticmethod
    def backward(ctx, grad_y):
        n_groups = len(ctx.groups)
        S_flat, trees, j = ctx.saved_tensors[:3]
        fallback = ctx.saved_tensors[3:3 + n_groups]
        margins = ctx.saved_tensors[3 + n_groups:]
        groups = ctx.groups
        n_bins = ctx.n_bins
        cosine = ctx.selection == SELECTION_COSINE
        trees64 = trees.to(torch.int64)

        grad_y = grad_y.contiguous()
        idx = trees64 * n_bins + j
        need_S = ctx.needs_input_grad[0]
        need_margins = any(ctx.needs_input_grad[7:])

        grad_S = None
        if need_S:
            y_dim = grad_y.shape[-1]
            flat_idx = idx.reshape(-1)
            values = grad_y.unsqueeze(-2).expand(*idx.shape, y_dim).reshape(-1, y_dim)
            # One row of `values` per visit. Dense costs S_flat.shape[0] rows regardless;
            # sparse costs one row per visit, so it only wins while visits < rows.
            dense = (ctx.table_grad == TABLE_GRAD_DENSE
                     or (ctx.table_grad == TABLE_GRAD_AUTO
                         and flat_idx.numel() >= S_flat.shape[0]))
            if dense:
                grad_S = torch.zeros_like(S_flat).index_add_(0, flat_idx, values)
            else:
                # Indices are constructed, not user-supplied, so skip invariant checks
                # (the context manager form also keeps torch from warning about it).
                with torch.sparse.check_sparse_tensor_invariants(False):
                    grad_S = torch.sparse_coo_tensor(
                        flat_idx.unsqueeze(0), values, S_flat.shape,
                        check_invariants=False,
                    )

        grad_margins = [None] * len(margins)
        if need_margins:
            S_j = F.embedding(idx, S_flat)
            grad_yt = grad_y.unsqueeze(-2)
            grad_sq = (grad_y * grad_y).sum(dim=-1, keepdim=True) if cosine else None
            # gi = (S[jbar] - S[j]) . g  is split so the S[j] half is contracted once for
            # all groups, and each contraction is a batched mat-vec rather than an
            # elementwise product plus reduction over a (B, P, N_T, y_dim) temporary.
            grad_col = grad_y.unsqueeze(-1)
            S_j_dot = torch.matmul(S_j, grad_col).squeeze(-1)

            for gi_idx, (group, margin) in enumerate(zip(groups, margins)):
                if not ctx.needs_input_grad[7 + gi_idx]:
                    continue
                if cosine:
                    grad_margins[gi_idx] = _cosine_group_grad(
                        S_flat, trees64, n_bins, j, S_j, grad_yt, grad_sq, idx,
                        margin, group, fallback[gi_idx])
                else:
                    grad_margins[gi_idx] = _closest_group_grad(
                        S_flat, trees64, n_bins, j, S_j_dot, grad_col,
                        margin, group, fallback[gi_idx])

        return (grad_S, None, None, None, None, None, None, *grad_margins)


def _closest_group_grad(S_flat, trees64, n_bins, j, S_j_dot, grad_col,
                        margin, group, r):
    """Margin gradient for ``closest`` selection.

    The chosen comparison is ``argmin |u|``, which depends only on the margins, so it is
    the same for every causal pair that shares a source position and (for a tree-broadcast
    group) for every tree.  That lets the whole gradient be assembled at the *source*
    shape: no ``(B, n_pairs, N_T, R)`` buffer is ever allocated.
    """
    r_pair = r if group.pair_index is None else r.index_select(1, group.pair_index)
    jbar = j ^ _flip_mask(r_pair, group)
    S_jbar = F.embedding(trees64 * n_bins + jbar, S_flat)
    gi = torch.matmul(S_jbar, grad_col).squeeze(-1) - S_j_dot            # (B, P, N_T)

    u = margin.gather(-1, r.unsqueeze(-1)).squeeze(-1)                   # source shape
    if group.pair_index is not None:
        u = u.index_select(1, group.pair_index)
    contribution = gi * _surrogate_factor(u)                             # (B, P, N_T)

    if group.pair_index is not None:
        accumulated = contribution.new_zeros(
            margin.shape[0], margin.shape[1], contribution.shape[-1])
        accumulated.index_add_(1, group.pair_index, contribution)
    else:
        accumulated = contribution
    if margin.shape[2] == 1:                 # margins shared across trees
        accumulated = accumulated.sum(dim=-1, keepdim=True)

    grad = torch.zeros_like(margin)
    return grad.scatter_(-1, r.unsqueeze(-1), accumulated.unsqueeze(-1))


def _cosine_group_grad(S_flat, trees64, n_bins, j, S_j, grad_yt, grad_sq, idx,
                       margin, group, r_fallback):
    """Margin gradient for ``cosine-distance`` selection.

    Here the choice of comparison depends on the upstream gradient and on the table rows,
    so it varies per pair and per tree; the gradient must be assembled at the expanded
    shape and scattered back.
    """
    gathered = _gather_pairs(margin, group).expand(*idx.shape, group.raw_count)
    r = _select_cosine(S_flat, trees64, n_bins, j, S_j, grad_yt, grad_sq,
                       gathered, group, _gather_pairs(r_fallback, group))

    jbar = j ^ _flip_mask(r, group)
    delta = F.embedding(trees64 * n_bins + jbar, S_flat) - S_j
    gi = (delta * grad_yt).sum(dim=-1)
    u = gathered.gather(-1, r.unsqueeze(-1)).squeeze(-1)

    grad_full = torch.zeros_like(gathered)
    grad_full.scatter_(-1, r.unsqueeze(-1), (gi * _surrogate_factor(u)).unsqueeze(-1))

    if group.pair_index is None:
        return _reduce_to(grad_full, margin.shape)
    reduced = _reduce_to(
        grad_full,
        torch.Size((margin.shape[0], grad_full.shape[1]) + margin.shape[2:]))
    scattered = torch.zeros_like(margin)
    scattered.index_add_(1, group.pair_index, reduced)
    return scattered


def _select_cosine(S_flat, trees64, n_bins, j, S_j, grad_yt, grad_sq,
                   gathered, group, r_fallback):
    """Replicate ``main._select_lut_pair_numba`` / ``_select_margin_candidate_python``.

    Scores each raw comparison by ``|delta . g| / sqrt(delta^2 * g^2) / (|u| + eps)`` and
    keeps the best, breaking ties on smaller ``|u|`` and then on smaller ``r``.  Candidates
    with ``delta^2 == 0`` or ``delta . g == 0`` are skipped, and when nothing is eligible
    (including ``|g|^2 == 0``) the closest-margin fallback is used.

    Iterates over raw comparisons rather than vectorising over them: ``R`` is small (6-18)
    and a vectorised form would materialise a ``(B, P, N_T, R, y_dim)`` tensor.
    """
    n_raw = group.raw_count
    best_score = torch.full(j.shape, -1.0, dtype=S_flat.dtype, device=S_flat.device)
    best_abs_u = torch.full(j.shape, float('inf'), dtype=S_flat.dtype, device=S_flat.device)
    best_r = torch.full(j.shape, -1, dtype=torch.int64, device=S_flat.device)

    for r in range(n_raw):
        bit = group.bit_offset + r // group.fan_in
        jbar = j ^ (1 << bit)
        delta = F.embedding(trees64 * n_bins + jbar, S_flat) - S_j
        dot = (delta * grad_yt).sum(dim=-1)
        delta_sq = (delta * delta).sum(dim=-1)
        u = gathered[..., r]
        abs_u = u.abs()

        valid = (delta_sq != 0) & (dot != 0) & (grad_sq > 0)
        cosine = dot.abs() / torch.sqrt(delta_sq * grad_sq)
        score = cosine / (abs_u + COSINE_DISTANCE_EPS)

        better = valid & (
            (score > best_score)
            | ((score == best_score) & (abs_u < best_abs_u))
        )
        best_score = torch.where(better, score, best_score)
        best_abs_u = torch.where(better, abs_u, best_abs_u)
        best_r = torch.where(better, torch.full_like(best_r, r), best_r)

    return torch.where(best_r >= 0, best_r, r_fallback)


def lut_lookup(S_flat, trees, n_bins, groups, margins,
               selection=SELECTION_CLOSEST, table_grad=TABLE_GRAD_AUTO, capture=None):
    """Look up and sum LUT rows. Returns ``(B, P, y_dim)``.

    Args:
        S_flat: ``(n_physical_trees * n_bins, y_dim)`` table.
        trees: ``(N_T,)`` map from logical comparator slot to physical tree
            (``--weight-sharing`` makes this non-injective).
        n_bins: ``2 ** total_n_c``.
        groups: one :class:`LutGroup` per margin source.
        margins: matching margin tensors, ``(B, P_src, N_T or 1, raw_count)``.
        selection: ``'closest'`` or ``'cosine-distance'``.
        table_grad: ``'auto'`` (dense once visits reach the table's row count),
            ``'dense'``, or ``'sparse'``. ``'auto'`` is almost always right; force
            ``'sparse'`` only for a table so large that a dense gradient will not fit.
        capture: optional dict; receives ``j`` (the row index, ``(B, P, N_T)``) and
            ``r_closest`` (per group, the closest-margin comparison at the margin
            source's own shape) for instrumentation.
    """
    return LUTFunction.apply(S_flat, trees, n_bins, tuple(groups), selection,
                             table_grad, capture, *margins)


def lut_lookup_reference(S_flat, trees, n_bins, groups, margins, capture=None):
    """Pure-autograd equivalent of :func:`lut_lookup` for ``closest`` selection.

    Uses the identity ``dphi/du = -0.5*sign(u)/(1+|u|)^2`` for ``phi(u) = 0.5/(1+|u|)``:
    the added term is identically zero in the forward pass and carries exactly the
    surrogate gradient in backward.  Kept as a test oracle -- it cannot express
    ``cosine-distance``, whose selection depends on the upstream gradient.
    """
    n_trees = trees.shape[0]
    trees64 = trees.to(torch.int64)

    with torch.no_grad():
        j, fallback = _index_and_fallback([m.detach() for m in margins], groups, n_trees)

    idx = trees64 * n_bins + j
    y = F.embedding(idx, S_flat, sparse=True).sum(dim=-2)
    S_j = F.embedding(idx, S_flat).detach()

    for margin, group, r in zip(margins, groups, fallback):
        gathered = _gather_pairs(margin, group).expand(*idx.shape, group.raw_count)
        r = _gather_pairs(r, group).expand(idx.shape)
        u = gathered.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        phi = 0.5 / (1.0 + torch.where(u > 0, u, -u))
        jbar = j ^ _flip_mask(r, group)
        delta = (F.embedding(trees64 * n_bins + jbar, S_flat).detach() - S_j)
        y = y + (delta * (phi - phi.detach()).unsqueeze(-1)).sum(dim=-2)

    if capture is not None:
        capture['j'] = j
        capture['r_closest'] = fallback
    return y
