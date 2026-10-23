"""Model modules: LUT tables, the attention heads, and the full LUT model.

Construction deliberately follows ``main.build_Model``'s RNG draw order exactly -- Python's
``random`` for comparison pairs, NumPy's for the embedder and PE -- so that seeding both
implementations identically produces identical tables.  ``test_torch_parity.py`` depends on
this; see the note in :mod:`lut_torch.comparisons`.

Everything downstream of a margin tensor is plain torch, which is the point of the port:
QK normalisation, PE addition, direct-sign slicing and the Q/K/PE gradient routing that
``main.new_attention_backward`` writes out by hand all fall out of autograd here.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ATTENTION_HARD, OUTPUT_CODE_BINARY, OUTPUT_CODE_RANDOM
from .comparisons import (
    build_random_comparison_arrays,
    causal_pair_indices,
    final_tree_map,
    initialize_positional_encoding,
    random_vector,
)
from .functional import (
    GATE_CONDITIONAL,
    GATE_EXCLUSIVE,
    TABLE_GRAD_AUTO,
    LutGroup,
    lut_lookup,
    lut_lookup_reference,
)


def _parity(values, mask):
    """Parity of ``values & mask`` over GF(2), vectorised."""
    x = np.asarray(values, dtype=np.int64) & int(mask)
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def ecoc_masks(n_classes, n_bits, max_candidates=512):
    """Bit masks for a systematic linear code: ``K`` index bits then parity bits.

    Bit ``b`` of class ``k`` is ``parity(k & masks[b])``. The first
    ``K = ceil(log2 n_classes)`` masks are ``1, 2, 4, ...``, so those bits *are* the class
    index and codewords are distinct by construction. Every later mask is a parity check
    over a subset of the index bits -- a genuinely new coordinate, which is what
    zero-padding the index expansion failed to give.

    Because the code is linear, the distance between two codewords is the codeword weight
    of the XOR of their indices, so minimum distance is a minimum over at most
    ``2**K`` differences rather than over ``n_classes**2`` pairs. That makes an exact
    greedy choice affordable: at each step take the mask maximising (min distance, then
    mean distance). Candidates are capped at ``max_candidates`` (sampled with a *local*
    RNG, so the code is a deterministic function of ``(n_classes, n_bits)`` and does not
    depend on -- or perturb -- the run seed), which keeps this bounded at the vocabulary
    sizes where ``2**K`` is large.
    """
    k_bits = max(1, (n_classes - 1).bit_length())
    if n_bits < k_bits:
        raise ValueError(f'{n_bits} bits cannot index {n_classes} classes')
    masks = [1 << b for b in range(k_bits)]
    if n_bits == k_bits:
        return masks

    # Differences that actually occur between used class indices. For a linear code these
    # are the only messages whose codeword weight can be a pairwise distance.
    idx = np.arange(n_classes, dtype=np.int64)
    diffs = np.unique(np.bitwise_xor.outer(idx, idx))
    diffs = diffs[diffs != 0]

    weights = np.zeros(diffs.shape, dtype=np.int64)
    for mask in masks:
        weights += _parity(diffs, mask)

    rng = np.random.RandomState(0)
    pool = [m for m in range(1, 1 << k_bits) if m not in set(masks)]
    while len(masks) < n_bits:
        if not pool:  # every functional used; repeats are the only option left
            pool = [m for m in range(1, 1 << k_bits)]
        candidates = (pool if len(pool) <= max_candidates
                      else [pool[i] for i in
                            rng.choice(len(pool), max_candidates, replace=False)])
        best, best_key, best_bits = None, None, None
        for mask in candidates:
            bits = _parity(diffs, mask)
            w = weights + bits
            key = (int(w.min()), float(w.mean()), -mask)
            if best_key is None or key > best_key:
                best, best_key, best_bits = mask, key, bits
        masks.append(best)
        weights = weights + best_bits
        if best in pool:
            pool.remove(best)
    return masks


def binary_code_matrix(n_classes, n_bits):
    """A frozen ``+-1`` systematic ECOC, ``(n_classes, n_bits)``.

    At ``n_bits = ceil(log2 n_classes)`` this is exactly the binary expansion of the class
    index (the masks are ``1, 2, 4, ...``), so earlier results at that width still stand.
    Above it, the extra bits are parity checks chosen by :func:`ecoc_masks` rather than the
    zeros the index expansion would supply -- padding with the high bits of a small index
    gives constant columns, which only shift every logit equally and cancel in the softmax.

    Rows are scaled by ``1/sqrt(n_bits)`` so they are unit-norm, which puts this code on
    the same logit scale as the learned code's initialisation and makes the two comparable
    at a shared learning rate.

    Sign rather than ``0/1`` is not cosmetic: under a ``0/1`` code the readout can only
    add evidence, never subtract it. The two differ by ``sum_b y_b``, a per-position
    constant that cancels in the softmax, so they give the same distribution up to a
    factor of two on the logits -- but only the ``+-1`` form keeps that factor from
    depending on how many bits a class happens to have set.
    """
    idx = np.arange(n_classes, dtype=np.int64)
    bits = np.stack([_parity(idx, mask) for mask in ecoc_masks(n_classes, n_bits)],
                    axis=1)
    return torch.tensor((2 * bits - 1) / math.sqrt(n_bits), dtype=torch.float32)


def random_code_matrix(n_classes, n_bits, max_attempts=64):
    """A frozen i.i.d. ``+-1`` code, ``(n_classes, n_bits)``, rows unit-norm.

    This is the frozen code to use above ``ceil(log2 V)`` bits, because the binary
    expansion cannot use them: class indices ``0..17`` leave bits 5 and up clear, so
    ``binary_code_matrix(18, 10)`` has five constant columns and rank 6. Widening D under
    that construction buys exactly nothing.

    A random code does use every column, and that is the whole point of the extra bits:
    expected Hamming distance between two codewords is ``D/2`` rather than the binary
    code's minimum of 1, so no single coordinate's sign decides between two classes.

    Codewords must be distinct or two classes are literally indistinguishable, and drawing
    each one independently does not give that: 18 classes over the 32 codewords of a
    5-bit code collide with probability ~1, and even at 10 bits collisions are common. So
    this samples ``n_classes`` *distinct* codeword indices without replacement and expands
    them, rather than drawing bits and hoping.

    The draw is uniform over subsets but the index set is not built by permuting
    ``2**n_bits`` entries, which would be gigabytes once ``n_bits`` reaches the twenties --
    exactly the regime a full vocabulary needs.

    Above 62 bits the codeword index no longer fits an int64 and the sampler below cannot
    represent the space at all (``np.random.randint`` raises "high is out of bounds for
    int64"), so the bits are drawn directly and deduplicated by row instead. The two paths
    agree in distribution; the integer one is kept because it is what produced every code
    at D <= 62, and switching it would silently change those runs' codes.
    """
    if n_bits >= 63:
        bits = (np.random.randint(0, 2, size=(n_classes, n_bits))
                .astype(np.int64))
        # Distinctness is the contract. At this width a collision is vanishingly
        # unlikely, but "unlikely" is not "checked".
        while True:
            _, first = np.unique(bits, axis=0, return_index=True)
            dupes = np.setdiff1d(np.arange(n_classes), first)
            if dupes.size == 0:
                break
            bits[dupes] = np.random.randint(0, 2, size=(dupes.size, n_bits))
        return torch.tensor((2 * bits - 1) / math.sqrt(n_bits), dtype=torch.float32)

    if n_classes > 2 ** n_bits:
        raise ValueError(f'{n_bits} bits cannot give {n_classes} distinct codewords')
    space = 2 ** n_bits
    chosen, seen = [], set()
    while len(chosen) < n_classes:
        for value in np.random.randint(0, space, size=2 * (n_classes - len(chosen)) + 8):
            value = int(value)
            if value not in seen:
                seen.add(value)
                chosen.append(value)
                if len(chosen) == n_classes:
                    break
    codes = np.array(chosen, dtype=np.int64)[:, None]
    bits = (codes >> np.arange(n_bits, dtype=np.int64)[None, :]) & 1
    return torch.tensor((2 * bits - 1) / math.sqrt(n_bits), dtype=torch.float32)


def normalize_to_target(vec, target_std, eps):
    """Mirror of ``main._normalize_vector_to_target`` over the last dimension."""
    mean = vec.mean(dim=-1, keepdim=True)
    centered = vec - mean
    inv_std = torch.rsqrt((centered * centered).mean(dim=-1, keepdim=True) + eps)
    return target_std * centered * inv_std


class LUTTable(nn.Module):
    """One LUT: a table ``S``, its comparison pairs, and its logical-to-physical tree map.

    ``S`` is stored flat as ``(n_physical_trees * n_bins, y_dim)`` because that is the
    layout the lookup wants; :attr:`S` exposes the logical ``(trees, bins, y_dim)`` view
    that ``main.py`` uses.
    """

    def __init__(self, name, n_t, n_c, fan_in, total_n_c, y_dim, input_dim,
                 s_n_t=None, trees=None, tree_weights=False):
        super().__init__()
        self.name = name
        self.n_t = n_t
        self.n_c = n_c
        self.fan_in = fan_in
        self.total_n_c = total_n_c
        self.y_dim = y_dim
        self.n_bins = 1 << total_n_c
        self.s_n_t = n_t if s_n_t is None else s_n_t

        a_arr, b_arr = build_random_comparison_arrays(n_t, n_c * fan_in, input_dim)
        self.register_buffer('a_arr', torch.tensor(a_arr, dtype=torch.int64))
        self.register_buffer('b_arr', torch.tensor(b_arr, dtype=torch.int64))
        tree_map = np.arange(n_t, dtype=np.int32) if trees is None else np.asarray(trees)
        self.register_buffer('trees', torch.tensor(tree_map, dtype=torch.int64))
        self.S_flat = nn.Parameter(torch.zeros(self.s_n_t * self.n_bins, y_dim))
        # One weight per logical slot, so slots sharing a physical tree under
        # --weight-sharing still weigh separately. Ones: draws no RNG, changes nothing.
        self.tree_weights = nn.Parameter(torch.ones(n_t)) if tree_weights else None

    @property
    def S(self):
        """Logical ``(physical_trees, bins, y_dim)`` view of the table."""
        return self.S_flat.view(self.s_n_t, self.n_bins, self.y_dim)

    def margins(self, x):
        """``u = x[a] - x[b]`` for every tree, shape ``(..., N_T, n_c * fan_in)``."""
        return x[..., self.a_arr] - x[..., self.b_arr]

    def group(self, bit_offset=0, pair_index=None):
        return LutGroup(n_bits=self.n_c, fan_in=self.fan_in,
                        bit_offset=bit_offset, pair_index=pair_index)

    def forward(self, x, selection, capture=None, table_grad=TABLE_GRAD_AUTO):
        entry = {} if capture is not None else None
        y = lut_lookup(self.S_flat, self.trees, self.n_bins, [self.group()],
                       [self.margins(x)], selection=selection,
                       table_grad=table_grad, capture=entry,
                       tree_weights=self.tree_weights)
        if capture is not None:
            capture[self.name] = entry
        return y


class NewAttentionHead(nn.Module):
    """One ``--new-attention`` head: Q, K, V, Final and a positional encoding.

    Q/K produce ``attn_qk_dim`` vectors that are combined per causal pair and compared
    against (or added to) the PE to form the low bits of the Final row index; V produces
    the value vector whose comparisons form the high bits.  Only pairs with ``j < i``
    exist, so position 0 receives no attention output.
    """

    def __init__(self, cfg, name, frozen_pe=None):
        super().__init__()
        self.cfg = cfg
        self.name = name
        value_dim = cfg.head_value_dim

        # Draw order below mirrors main.build_Model: PE, then Q, K, V, Final, then the
        # QKPE comparison arrays (which are drawn even under --qkpe-direct-sign).
        if frozen_pe is not None:
            pe = np.array(frozen_pe, dtype=np.float32, copy=True)
        else:
            pe = random_vector(cfg.context_size * cfg.attn_qk_dim, 1.0).reshape(
                cfg.context_size, cfg.attn_qk_dim)
        self.positional_encoding = nn.Parameter(
            torch.tensor(pe, dtype=torch.float32), requires_grad=not cfg.freeze_PE)

        tw = cfg.tree_weights
        self.Q = LUTTable(f'{name}_Q', cfg.attn_qk_n_t, cfg.attn_qk_n_c,
                          cfg.comparisons_per_bit, cfg.attn_qk_n_c, cfg.attn_qk_dim,
                          cfg.embedding_dim, tree_weights=tw)
        self.K = LUTTable(f'{name}_K', cfg.attn_qk_n_t, cfg.attn_qk_n_c,
                          cfg.comparisons_per_bit, cfg.attn_qk_n_c, cfg.attn_qk_dim,
                          cfg.embedding_dim, tree_weights=tw)
        self.V = LUTTable(f'{name}_V', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                          cfg.n_c, value_dim, cfg.embedding_dim, tree_weights=tw)
        self.Final = LUTTable(f'{name}_Final', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                              cfg.final_total_n_c, value_dim, value_dim,
                              s_n_t=cfg.final_n_t,
                              trees=final_tree_map(cfg.final_n_t, cfg.final_share_m),
                              tree_weights=tw)

        qkpe_a, qkpe_b = build_random_comparison_arrays(
            cfg.n_t, cfg.positional_dim, cfg.attn_qk_dim)
        self.register_buffer('qkpe_a', torch.tensor(qkpe_a, dtype=torch.int64))
        self.register_buffer('qkpe_b', torch.tensor(qkpe_b, dtype=torch.int64))

        tok_i, tok_j = causal_pair_indices(cfg.context_size)
        self.register_buffer('tok_i', torch.tensor(tok_i))
        self.register_buffer('tok_j', torch.tensor(tok_j))
        self.register_buffer('pe_pos', torch.tensor(tok_i - tok_j))

        # CSR inverse of tok_j (pair -> source position), for the fused-kernel backend's
        # per-source-position parallel backward (lut_torch.kernels._k_grad_paired). Built
        # once here rather than per call; unused by the torch backward path.
        tok_j_t = self.tok_j
        order = torch.argsort(tok_j_t, stable=True)
        counts = torch.bincount(tok_j_t, minlength=cfg.context_size)
        pair_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        self.register_buffer('final_pair_ptr', pair_ptr)
        self.register_buffer('final_pair_list', order)

    def qkpe_margins(self, x, selection, capture):
        """Margins for the positional/low bits of the Final row index."""
        cfg = self.cfg
        active = cfg.qkpe_active_dim
        pe = self.positional_encoding.index_select(0, self.pe_pos)[..., :active]

        if cfg.PE_only:
            combined = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            q = self.Q(x, selection, capture, cfg.table_grad)
            k = self.K(x, selection, capture, cfg.table_grad)
            qk = q.index_select(1, self.tok_i) + k.index_select(1, self.tok_j)
            qk = qk[..., :active]
            if cfg.normalize_qk:
                qk = normalize_to_target(qk, cfg.qk_norm_target_std, cfg.qk_norm_eps)
            combined = qk + pe

        if cfg.qkpe_direct_sign:
            # Bit c is just sign(combined[c]); the margin is shared by every tree, so
            # keep the tree axis at size 1 and let the lookup broadcast it.
            return combined[..., :cfg.positional_dim].unsqueeze(-2)
        return combined[..., self.qkpe_a] - combined[..., self.qkpe_b]

    def forward(self, x, selection, capture=None):
        cfg = self.cfg
        v = self.V(x, selection, capture, cfg.table_grad)
        qkpe = self.qkpe_margins(x, selection, capture)
        v_margins = self.Final.margins(v)

        groups = [
            LutGroup(n_bits=cfg.positional_dim, fan_in=1, bit_offset=0),
            LutGroup(n_bits=self.Final.n_c, fan_in=self.Final.fan_in,
                    bit_offset=cfg.positional_dim, pair_index=self.tok_j,
                    pair_ptr=self.final_pair_ptr, pair_list=self.final_pair_list),
        ]
        entry = {} if capture is not None else None
        y_pairs = lut_lookup(self.Final.S_flat, self.Final.trees, self.Final.n_bins,
                             groups, [qkpe, v_margins], selection=selection,
                             table_grad=cfg.table_grad, capture=entry,
                             tree_weights=self.Final.tree_weights)
        if capture is not None:
            capture[f'{self.name}_Final'] = entry

        out = x.new_zeros(x.shape[0], cfg.context_size, cfg.head_value_dim)
        return out.index_add_(1, self.tok_i, y_pairs)


class HardAttentionHead(nn.Module):
    """``main.py``'s original hard attention head (``AttentionHead.V``).

    One table per head, indexed per causal pair ``(i, j < i)`` and per tree by::

        row = jV[i] << (n_c + pd)  |  jV[j] << pd  |  jPE[i - j]

    ``jV`` is the head's own comparison bits of the layer input, read at the query position
    ``i`` and at the key position ``j`` -- one set of comparisons serves both sides.
    ``jPE`` is the sign pattern of a learned ``(CS, N_T, pd)`` positional encoding at the
    relative offset. The output at ``i`` is the plain sum over every ``j < i`` (no softmax,
    no normalisation), so position 0 receives nothing.

    The gradient is not three independent groups. ``main._attention_backward_kernel``
    flips only the query bit *or* the key bit, whichever closest margin is smaller (ties to
    the key), and flips the PE bit in addition only when its margin is smaller than both.
    That is the ``exclusive`` / ``conditional`` gating in :mod:`lut_torch.functional`, with
    the key group listed before the query group so first-minimum tie-breaking matches.

    Extensions with no ``main.py`` counterpart (``test_torch_parity.py`` cannot check
    them): ``--new-MHA`` narrows the table to ``embedding_dim / num_heads``,
    ``--weight-sharing`` shares its physical trees as it does Final's, and
    ``--hard-attn-n-c`` decouples its per-side bits from ``n_c``.
    """

    def __init__(self, cfg, name, frozen_pe=None):
        super().__init__()
        self.cfg = cfg
        self.name = name
        n_bits = cfg.hard_attn_bits

        # Draw order mirrors main.build_Model's hard branch: PE, then the table.
        if frozen_pe is not None:
            pe = np.broadcast_to(np.asarray(frozen_pe, dtype=np.float32)[:, None, :],
                                 (cfg.context_size, cfg.n_t, cfg.positional_dim)).copy()
        else:
            pe = random_vector(cfg.context_size * cfg.n_t * cfg.positional_dim,
                               1.0).reshape(cfg.context_size, cfg.n_t, cfg.positional_dim)
        self.positional_encoding = nn.Parameter(
            torch.tensor(pe, dtype=torch.float32), requires_grad=not cfg.freeze_PE)

        self.V = LUTTable(f'{name}_V', cfg.n_t, n_bits, cfg.comparisons_per_bit,
                          cfg.hard_attn_total_n_c, cfg.head_value_dim, cfg.embedding_dim,
                          s_n_t=cfg.final_n_t,
                          trees=final_tree_map(cfg.final_n_t, cfg.final_share_m),
                          tree_weights=cfg.tree_weights)

        tok_i, tok_j = causal_pair_indices(cfg.context_size)
        self.register_buffer('tok_i', torch.tensor(tok_i))
        self.register_buffer('tok_j', torch.tensor(tok_j))
        self.register_buffer('pe_pos', torch.tensor(tok_i - tok_j))

    def forward(self, x, selection, capture=None):
        cfg = self.cfg
        n_bits = self.V.n_c
        pd = cfg.positional_dim
        margins = self.V.margins(x)                               # (B, CS, N_T, R)
        pe = self.positional_encoding.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

        groups = [
            LutGroup(n_bits=n_bits, fan_in=self.V.fan_in, bit_offset=pd,
                     pair_index=self.tok_j, gate=GATE_EXCLUSIVE),            # key
            LutGroup(n_bits=n_bits, fan_in=self.V.fan_in, bit_offset=n_bits + pd,
                     pair_index=self.tok_i, gate=GATE_EXCLUSIVE),            # query
            LutGroup(n_bits=pd, fan_in=1, bit_offset=0,
                     pair_index=self.pe_pos, gate=GATE_CONDITIONAL),         # PE
        ]
        entry = {} if capture is not None else None
        y_pairs = lut_lookup(self.V.S_flat, self.V.trees, self.V.n_bins, groups,
                             [margins, margins, pe], selection=selection,
                             table_grad=cfg.table_grad, capture=entry,
                             tree_weights=self.V.tree_weights)
        if capture is not None:
            capture[self.V.name] = entry

        out = x.new_zeros(x.shape[0], cfg.context_size, cfg.head_value_dim)
        return out.index_add_(1, self.tok_i, y_pairs)


class StackedHeads(nn.Module):
    """All of a layer's attention heads, executed as 4 LUT calls instead of ``4 * H``.

    The heads of a layer are mutually independent and read the same ``x``; Q and K on top
    of that have identical shapes.  Running them as separate :class:`NewAttentionHead`
    modules therefore costs ``H`` times the *calls* for the same work, and at batch 1 this
    model is bound by call count, not work.  Measured on an RTX 2080 Ti at the production
    POS shape: 8 separate Final calls cost 15.07 ms while one call with 8x the rows costs
    2.63 ms, and a single-head call already costs 2.46 ms -- eight times the work for 7%
    more time.  A single small LUT call occupies 1.5% of the card's 69,632 resident
    threads, so the machine is idle waiting for launches, not busy.

    The stacking axis is folded into the *batch* dimension, and each stacked table gets its
    own slab of one shared ``S_flat``; which slab is selected by the per-row tree map that
    :func:`lut_torch.functional._tree_rows` accepts.  Nothing else about the computation
    changes, so the **forward is bit-identical** to the unstacked path -- including the
    order in which head outputs are summed back into the residual stream, which
    :class:`LutLayer` deliberately keeps sequential for exactly that reason.

    The backward is bit-identical too *except* for the gradient that flows back into the
    layer's input.  Unstacked, ``u = x[a] - x[b]`` is ``H`` separate gathers whose
    backwards are ``H`` separate scatter-adds into ``x.grad``; stacked, it is one gather
    and one scatter-add covering every head, so the same terms are summed in a different
    order.  Float32 addition is not associative, so layers *below* the top see a
    difference of 1-3 ULP (measured: 0.0 at one layer, 2.1e-7 relative at two, 3.2e-7 at
    three).  That is the same class of reassociation as changing a thread count, it does
    not compound with depth, and closing it would mean giving up the fused gather -- the
    interleaving of per-head V/Q/K gathers cannot be reproduced once the heads share a
    call.  ``test_torch_stacking.py`` pins both halves of this: exact forward, exact
    single-layer gradients, and a float32-epsilon bound on deeper ones.

    Construction goes through real :class:`NewAttentionHead` objects and then stacks their
    buffers.  That is not laziness: it is the only way to guarantee the RNG draw order
    still matches ``main.build_Model`` head by head, which is what
    ``test_torch_parity.py`` rests on.  The per-head tables are all-zero at construction,
    so nothing but small index buffers is copied.
    """

    def __init__(self, cfg, layer_index, frozen_pe=None):
        super().__init__()
        self.cfg = cfg
        heads = [NewAttentionHead(cfg, f'head_{layer_index}_{h}', frozen_pe)
                 for h in range(cfg.num_heads)]
        self.head_names = [h.name for h in heads]
        H = cfg.num_heads

        def stack(attr, mods):
            return torch.stack([getattr(m, attr) for m in mods])

        # Q and K share a shape, so they stack into one table of 2H slots: slots [0, H)
        # are the Q heads and [H, 2H) the K heads.
        qk = [h.Q for h in heads] + [h.K for h in heads]
        self._add_stack('qk', qk, cfg.attn_qk_dim)
        self._add_stack('v', [h.V for h in heads], cfg.head_value_dim)
        self._add_stack('final', [h.Final for h in heads], cfg.head_value_dim)

        self.positional_encoding = nn.Parameter(
            torch.stack([h.positional_encoding.detach() for h in heads]),
            requires_grad=not cfg.freeze_PE)
        self.register_buffer('qkpe_a', stack('qkpe_a', heads))
        self.register_buffer('qkpe_b', stack('qkpe_b', heads))

        first = heads[0]
        for name in ('tok_i', 'tok_j', 'pe_pos', 'final_pair_ptr', 'final_pair_list'):
            self.register_buffer(name, getattr(first, name).clone())
        self.n_pairs = self.tok_i.shape[0]

    def _add_stack(self, prefix, tables, y_dim):
        """Register one stacked table: flat parameter, flat a/b indices, offset tree map."""
        n_slots = len(tables)
        first = tables[0]
        rows_per_slot = first.s_n_t * first.n_bins
        setattr(self, f'{prefix}_n_bins', first.n_bins)
        setattr(self, f'{prefix}_n_c', first.n_c)
        setattr(self, f'{prefix}_fan_in', first.fan_in)
        setattr(self, f'{prefix}_n_t', first.n_t)
        setattr(self, f'{prefix}_slots', n_slots)
        self.register_parameter(
            f'{prefix}_S_flat',
            nn.Parameter(torch.zeros(n_slots * rows_per_slot, y_dim)))
        # a/b flattened to a single index vector so the margins are one index_select
        # rather than one per slot; the (slot, tree, raw) structure is restored by a view.
        self.register_buffer(f'{prefix}_a',
                             torch.stack([t.a_arr for t in tables]))
        self.register_buffer(f'{prefix}_b',
                             torch.stack([t.b_arr for t in tables]))
        # Each slot's rows sit at slot * rows_per_slot, so the tree map carries the offset
        # and one call addresses every slot's own table.
        offsets = (torch.arange(n_slots, dtype=torch.int64).unsqueeze(1)
                   * first.s_n_t)
        self.register_buffer(f'{prefix}_trees',
                             torch.stack([t.trees for t in tables]) + offsets)
        # (slots, N_T): row s is slot s's own weights, as in the unstacked tables.
        self.register_parameter(
            f'{prefix}_tree_weights',
            nn.Parameter(torch.stack([t.tree_weights.detach() for t in tables]))
            if first.tree_weights is not None else None)

    def _slot_trees(self, prefix, batch):
        """The stacked tree map replicated over the batch, as ``(batch * slots, N_T)``."""
        trees = getattr(self, f'{prefix}_trees')
        slots, n_t = trees.shape
        return trees.unsqueeze(0).expand(batch, slots, n_t).reshape(batch * slots, n_t)

    def _slot_tree_weights(self, prefix, batch):
        """The stacked tree weights replicated like :meth:`_slot_trees`, or ``None``."""
        weights = getattr(self, f'{prefix}_tree_weights')
        if weights is None:
            return None
        slots, n_t = weights.shape
        return weights.unsqueeze(0).expand(batch, slots, n_t).reshape(batch * slots, n_t)

    def _shared_margins(self, prefix, x):
        """``u = x[a] - x[b]`` for every slot, from an input all slots share.

        Returns ``(B * slots, P, N_T, R)``.  One ``index_select`` covers every slot; the
        permute that follows moves the slot axis next to the batch so the fold is
        ``b * slots + s``.
        """
        a = getattr(self, f'{prefix}_a')
        b = getattr(self, f'{prefix}_b')
        slots, n_t, raw = a.shape
        flat_a, flat_b = a.reshape(-1), b.reshape(-1)
        u = x.index_select(-1, flat_a) - x.index_select(-1, flat_b)
        u = u.view(*x.shape[:-1], slots, n_t, raw)
        return u.permute(0, 2, 1, 3, 4).reshape(x.shape[0] * slots, x.shape[1], n_t, raw)

    def _per_slot_margins(self, prefix, x):
        """``u = x[a] - x[b]`` where both the input and the indices are per slot.

        ``x`` is ``(B, slots, P, D)`` -- the Final table's V input.  Returns
        ``(B * slots, P, N_T, R)``; no permute is needed, the slot axis is already in
        place.
        """
        a = getattr(self, f'{prefix}_a')
        b = getattr(self, f'{prefix}_b')
        slots, n_t, raw = a.shape
        B, _, P, _ = x.shape
        shape = (B, slots, P, n_t * raw)
        ua = x.gather(-1, a.reshape(1, slots, 1, -1).expand(shape))
        ub = x.gather(-1, b.reshape(1, slots, 1, -1).expand(shape))
        return (ua - ub).view(B * slots, P, n_t, raw)

    def _lookup(self, prefix, margins, groups, batch, capture, names):
        table_grad = self.cfg.table_grad
        entry = {} if capture is not None else None
        y = lut_lookup(getattr(self, f'{prefix}_S_flat'), self._slot_trees(prefix, batch),
                       getattr(self, f'{prefix}_n_bins'), groups, margins,
                       selection=self.cfg.gradient_pair_selection,
                       table_grad=table_grad, capture=entry,
                       tree_weights=self._slot_tree_weights(prefix, batch))
        if capture is not None:
            self._split_capture(entry, capture, names, batch, len(names))
        return y

    def _split_capture(self, entry, capture, names, batch, slots):
        """Re-expose a stacked call's ``j``/``r`` per slot, under the unstacked names.

        The slab offset lives in the tree map, not in ``j``, so a slot's rows are exactly
        the unstacked model's -- a plain reshape and select recovers them.
        """
        j = entry['j'].view(batch, slots, *entry['j'].shape[1:])
        for slot, name in enumerate(names):
            capture[name] = {
                'j': j[:, slot],
                'r_closest': [r.view(batch, slots, *r.shape[1:])[:, slot]
                              for r in entry['r_closest']],
            }

    def forward(self, x, selection, capture=None):
        cfg = self.cfg
        B, CS, _ = x.shape
        H = cfg.num_heads

        # --- V, and Q/K together: three stacked calls covering 3H tables -------------
        v_margins = self._shared_margins('v', x)
        v = self._lookup('v', [v_margins],
                         [LutGroup(n_bits=self.v_n_c, fan_in=self.v_fan_in,
                                   bit_offset=0)],
                         B, capture, [f'{n}_V' for n in self.head_names])
        v = v.view(B, H, CS, cfg.head_value_dim)

        if cfg.PE_only:
            qkpe = self._pe_only_margins(x, B)
        else:
            qk_margins = self._shared_margins('qk', x)
            qk = self._lookup('qk', [qk_margins],
                              [LutGroup(n_bits=self.qk_n_c, fan_in=self.qk_fan_in,
                                        bit_offset=0)],
                              B, capture,
                              [f'{n}_Q' for n in self.head_names]
                              + [f'{n}_K' for n in self.head_names])
            qk = qk.view(B, 2 * H, CS, cfg.attn_qk_dim)
            qkpe = self._qkpe_margins(qk[:, :H], qk[:, H:], B)

        # --- Final: one call covering all H heads -----------------------------------
        final_margins = self._per_slot_margins('final', v)
        groups = [
            LutGroup(n_bits=cfg.positional_dim, fan_in=1, bit_offset=0),
            LutGroup(n_bits=self.final_n_c, fan_in=self.final_fan_in,
                     bit_offset=cfg.positional_dim, pair_index=self.tok_j,
                     pair_ptr=self.final_pair_ptr, pair_list=self.final_pair_list),
        ]
        y_pairs = self._lookup('final', [qkpe, final_margins], groups, B, capture,
                               [f'{n}_Final' for n in self.head_names])
        y_pairs = y_pairs.view(B, H, self.n_pairs, cfg.head_value_dim)

        out = x.new_zeros(B, H, CS, cfg.head_value_dim)
        return out.index_add_(2, self.tok_i, y_pairs)

    def _pe(self, active):
        return self.positional_encoding.index_select(1, self.pe_pos)[..., :active]

    def _pe_only_margins(self, x, batch):
        cfg = self.cfg
        combined = self._pe(cfg.qkpe_active_dim).unsqueeze(0).expand(
            batch, -1, -1, -1)
        return self._finish_qkpe(combined, batch)

    def _qkpe_margins(self, q, k, batch):
        cfg = self.cfg
        active = cfg.qkpe_active_dim
        qk = q.index_select(2, self.tok_i) + k.index_select(2, self.tok_j)
        qk = qk[..., :active]
        if cfg.normalize_qk:
            qk = normalize_to_target(qk, cfg.qk_norm_target_std, cfg.qk_norm_eps)
        return self._finish_qkpe(qk + self._pe(active), batch)

    def _finish_qkpe(self, combined, batch):
        """``combined`` is ``(B, H, n_pairs, active)`` -> margins ``(B*H, n_pairs, T, R)``."""
        cfg = self.cfg
        H = cfg.num_heads
        if cfg.qkpe_direct_sign:
            # Bit c is sign(combined[c]); the margin is shared by every tree, so the tree
            # axis stays at size 1 and the lookup broadcasts it.
            out = combined[..., :cfg.positional_dim].unsqueeze(-2)
            return out.reshape(batch * H, self.n_pairs, 1, cfg.positional_dim)
        a, b = self.qkpe_a, self.qkpe_b
        slots, n_t, raw = a.shape
        shape = combined.shape[:-1] + (n_t * raw,)
        ua = combined.gather(-1, a.reshape(1, slots, 1, -1).expand(shape))
        ub = combined.gather(-1, b.reshape(1, slots, 1, -1).expand(shape))
        return (ua - ub).reshape(batch * H, self.n_pairs, n_t, raw)


class LutLayer(nn.Module):
    """Pre-norm attention block plus pre-norm FFN block, both residual."""

    def __init__(self, cfg, index, frozen_pe=None):
        super().__init__()
        self.cfg = cfg
        ed = cfg.embedding_dim

        self.ffn = LUTTable(f'ffn_{index}', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                            cfg.n_c, ed, ed, tree_weights=cfg.tree_weights)
        if cfg.new_MHA:
            self.mha_o = LUTTable(f'mha_o_{index}', cfg.n_t, cfg.n_c,
                                  cfg.comparisons_per_bit, cfg.n_c, ed,
                                  cfg.num_heads * cfg.head_value_dim,
                                  tree_weights=cfg.tree_weights)
        else:
            self.mha_o = None
        self.stacked = cfg.stack_heads
        if self.stacked:
            self.heads = StackedHeads(cfg, index, frozen_pe)
        else:
            head_cls = (HardAttentionHead if cfg.attention == ATTENTION_HARD
                        else NewAttentionHead)
            self.heads = nn.ModuleList([
                head_cls(cfg, f'head_{index}_{h}', frozen_pe)
                for h in range(cfg.num_heads)
            ])

        self.ln_attn_gamma = nn.Parameter(torch.ones(ed))
        self.ln_attn_beta = nn.Parameter(torch.zeros(ed))
        self.ln_ffn_gamma = nn.Parameter(torch.ones(ed))
        self.ln_ffn_beta = nn.Parameter(torch.zeros(ed))

    def _residual_dropout(self, z_pre, z):
        if self.cfg.dropout > 0.0 and self.training:
            return z_pre + F.dropout(z - z_pre, p=self.cfg.dropout, training=True)
        return z

    def forward(self, z, selection, capture=None):
        cfg = self.cfg
        ed = cfg.embedding_dim

        x = (F.layer_norm(z, (ed,), self.ln_attn_gamma, self.ln_attn_beta, eps=1e-5)
             if cfg.layernorm else z)
        z_pre = z
        if self.stacked:
            # (B, H, CS, head_value_dim) -- one call per table role, not per head.
            per_head = self.heads(x, selection, capture)
            if cfg.new_MHA:
                z = z + self.mha_o(
                    per_head.permute(0, 2, 1, 3).reshape(*x.shape[:2], -1),
                    selection, capture, cfg.table_grad)
            else:
                # Added one head at a time, in head order, rather than as a single
                # `per_head.sum(1)`: float32 addition is not associative, and keeping the
                # sequence identical to the unstacked path is what makes stacking
                # bit-exact rather than merely close (test_torch_stacking.py).
                for h in range(cfg.num_heads):
                    z = z + per_head[:, h]
        elif cfg.new_MHA:
            concat = torch.cat([h(x, selection, capture) for h in self.heads], dim=-1)
            z = z + self.mha_o(concat, selection, capture, cfg.table_grad)
        else:
            for head in self.heads:
                z = z + head(x, selection, capture)
        z = self._residual_dropout(z_pre, z)

        ffn_in = (F.layer_norm(z, (ed,), self.ln_ffn_gamma, self.ln_ffn_beta, eps=1e-5)
                  if cfg.layernorm else z)
        z_pre = z
        z = z + self.ffn(ffn_in, selection, capture, cfg.table_grad)
        return self._residual_dropout(z_pre, z)


class LutModel(nn.Module):
    """The full LUT model: frozen token embedding, N layers, and a LUT output head.

    The token embedding is a buffer rather than a parameter because ``main.py`` never
    updates it (the embedder update is commented out in ``model_backward``).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        embedder = random_vector(cfg.vocab_size_input * cfg.embedding_dim, 1.0).reshape(
            cfg.vocab_size_input, cfg.embedding_dim)
        self.register_buffer('token_embedding', torch.tensor(embedder,
                                                             dtype=torch.float32))

        frozen_pe = None
        if cfg.freeze_PE:
            # main.build_Model: hard attention freezes a (CS, positional_dim) pattern
            # shared by every tree; new attention one of width attn_qk_dim.
            pe_dim = (cfg.positional_dim if cfg.attention == ATTENTION_HARD
                      else cfg.attn_qk_dim)
            frozen_pe = initialize_positional_encoding(
                (cfg.context_size, pe_dim), cfg.PE_init_scale, cfg.positional_dim)

        self.layers = nn.ModuleList([
            LutLayer(cfg, l, frozen_pe) for l in range(cfg.num_layers)
        ])
        code_dim = cfg.output_code_dim
        self.output_head = LUTTable('output', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                                    cfg.n_c,
                                    cfg.vocab_size_output if code_dim is None
                                    else code_dim,
                                    cfg.embedding_dim, tree_weights=cfg.tree_weights)
        # Drawn last, and only when a code is asked for, so every existing configuration
        # leaves both RNG streams exactly where main.build_Model does.
        if code_dim is None:
            self.output_code = None
        elif cfg.output_code == OUTPUT_CODE_BINARY:
            self.register_buffer('output_code',
                                 binary_code_matrix(cfg.vocab_size_output, code_dim))
        elif cfg.output_code == OUTPUT_CODE_RANDOM:
            self.register_buffer('output_code',
                                 random_code_matrix(cfg.vocab_size_output, code_dim))
        else:
            # Unit-norm rows in expectation, matching binary_code_matrix. C must not
            # start at zero: y is zero at step 0 (S is), so dL/dC is zero too, and a
            # zero C would leave dL/dy zero as well -- neither the code nor the table
            # would ever receive a gradient.
            code = np.random.randn(cfg.vocab_size_output, code_dim) / math.sqrt(code_dim)
            self.output_code = nn.Parameter(torch.tensor(code, dtype=torch.float32))
        # Zero-initialised, so it draws no RNG and starts the run as a no-op.
        self.output_bias = (
            nn.Parameter(torch.zeros(cfg.vocab_size_output))
            if code_dim is not None and cfg.output_code_bias else None)

    def forward(self, tokens, capture=None):
        """``tokens``: ``(B, context_size)`` int64. Returns logits ``(B, CS, vocab)``.

        With ``output_code_dim`` set, the head emits ``D`` numbers per position and the
        logits are ``y @ C.T``. That linear readout *is* the per-bit-likelihood decoder:
        scoring class ``k`` by ``sum_b [c_kb log sigma(y_b) + (1-c_kb) log sigma(-y_b)]``
        gives ``sum_b c_kb * y_b`` plus ``sum_b log sigma(-y_b)``, and the second term
        does not depend on ``k``, so it cancels in the softmax. Treating the ``D``
        coordinates as independent bit comparisons and taking cross-entropy over the
        codewords is therefore exactly this matmul -- no sigmoids, and no straight-through
        estimator for the thresholding, which only ever happens at argmax time.
        """
        selection = self.cfg.gradient_pair_selection
        z = self.token_embedding[tokens]
        for layer in self.layers:
            z = layer(z, selection, capture)
        y = self.output_head(z, selection, capture, self.cfg.table_grad)
        if self.output_code is None:
            return y
        logits = y @ self.output_code.t()
        return logits if self.output_bias is None else logits + self.output_bias

    def table_parameters(self):
        """The table parameters (``S``), for optimizer grouping.

        A :class:`StackedHeads` block holds one table per role covering all its heads, so
        it contributes three parameters where the unstacked path contributes ``4 * H``.
        """
        out = []
        for module in self.modules():
            if isinstance(module, LUTTable):
                out.append(module.S_flat)
            elif isinstance(module, StackedHeads):
                out.extend([module.qk_S_flat, module.v_S_flat, module.final_S_flat])
        return out

    def tree_weight_parameters(self):
        """The per-tree weights (``--tree-weights``), for optimizer grouping."""
        out = []
        for module in self.modules():
            if isinstance(module, LUTTable):
                out.append(module.tree_weights)
            elif isinstance(module, StackedHeads):
                out.extend([module.qk_tree_weights, module.v_tree_weights,
                            module.final_tree_weights])
        return [p for p in out if p is not None and p.requires_grad]

    def output_code_parameters(self):
        """The coded head's dense parameters (``C`` and ``b``), for optimizer grouping.

        These are split out from :meth:`dense_parameters` because they sit on the wrong
        side of an asymmetry. A token's ``y`` is the sum of ``n_t`` table rows and *every*
        one of them receives the full ``dL/dy``, so one SGD step moves the table's
        contribution to the output by roughly ``n_t`` times a nominal step. ``C`` and ``b``
        have no such multiplier -- one step is one step. At a shared learning rate the
        table therefore adapts far faster and absorbs most of the learning. That costs
        little at POS, where ``C`` is 0.1% of the head, and a great deal at a full
        vocabulary, where ``C`` is about half of it. ``--output-code-lr-scale`` is the knob
        that lets the question be answered rather than assumed.
        """
        out = []
        if isinstance(self.output_code, nn.Parameter):
            out.append(self.output_code)
        if self.output_bias is not None:
            out.append(self.output_bias)
        return [p for p in out if p.requires_grad]

    def dense_parameters(self):
        """Everything else that trains: positional encodings and LayerNorm affines."""
        excluded = {id(p) for p in self.table_parameters()}
        excluded.update(id(p) for p in self.output_code_parameters())
        excluded.update(id(p) for p in self.tree_weight_parameters())
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in excluded]
