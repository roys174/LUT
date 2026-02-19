"""SNN transformer architecture in pure Python/NumPy."""
"""Based on Spiking Manifesto (Izhikevich 2025)"""
"""The code strives for simplicity, not efficiency."""
"""Eugene Izhikevich, October 2025"""
"""Python port by Claude, February 2026"""

import argparse
import io
import math
import random
import sys
import numpy as np
import tiktoken
from tqdm import tqdm

try:
    from numba import njit, prange, get_num_threads, get_thread_id
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range
    get_num_threads = lambda: 1
    get_thread_id = lambda: 0
    def njit(*args, **kwargs):
        def decorator(f): return f
        return decorator


# Global learning rate (updated each step)
learning_rate = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def softmax(x, size, temperature):
    max_val = np.max(x[:size])
    x[:size] = np.exp((x[:size] - max_val) / temperature)
    s = np.sum(x[:size])
    x[:size] /= s


def random_vector(size, scale):
    return scale * 2.0 * (np.random.rand(size).astype(np.float32) - 0.5)


def sample(probabilities, n):
    cumsum = np.cumsum(probabilities[:n])
    return min(int(np.searchsorted(cumsum, random.random())), n - 1)


def fill_vector_with_random_integers(N, max_value):
    return np.array([random.randint(0, max_value - 1) for _ in range(N)], dtype=np.int32)


def fill_vector_with_random_integers_different_from_vector2(vector2, N, max_value):
    vector = np.zeros(N, dtype=np.int32)
    for i in range(N):
        while True:
            vector[i] = random.randint(0, max_value - 1)
            if vector[i] != vector2[i]:
                break
    return vector


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class LUT:
    __slots__ = ['y_dim', 'S', 'a_arr', 'b_arr', 'bitmasks', 'trees']
    def __init__(self, n_t):
        self.y_dim = 0
        self.S = None
        self.a_arr = None
        self.b_arr = None
        self.bitmasks = None
        self.trees = np.arange(n_t, dtype=np.int32)


class AttentionHead:
    __slots__ = ['V', 'Positional_encoding',
                 '_v_j', '_v_r_min', '_v_u_min',
                 '_pe_j', '_pe_r_min', '_pe_u_min']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        POSITIONAL_DIM = args.positional_dim
        self.V = LUT(N_T)
        self.Positional_encoding = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)
        self._v_j = np.zeros((CS, N_T), dtype=np.int32)
        self._v_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._v_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._pe_j = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_u_min = np.zeros((CS, N_T), dtype=np.float32)


class StandardOutputHead:
    __slots__ = ['unembedder', 'output', 'tokens', '_j', '_r_min', '_u_min']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        VOCAB_SIZE = args.vocab_size
        self.unembedder = LUT(N_T)
        self.output = np.zeros((CS, VOCAB_SIZE), dtype=np.float32)
        self.tokens = None  # set by load_targets
        self._j = np.zeros((CS, N_T), dtype=np.int32)
        self._r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._u_min = np.zeros((CS, N_T), dtype=np.float32)

    def build(self, n_c, args):
        build_LUT(self.unembedder, n_c, args.vocab_size, args)

    def load_targets(self, tokens, context_size):
        self.tokens = tokens

    def forward(self, z, args):
        self.output[:] = 0.0
        cache_index_batch(self.unembedder, z, self._j, self._r_min, self._u_min)
        lut = self.unembedder
        self.output[:, :lut.y_dim] += _f32(lut.S[lut.trees, self._j]).sum(axis=1)

    def backward(self, x_grad, args):
        global learning_rate
        lut = self.unembedder
        if HAS_NUMBA and lut.S.dtype == np.float32:
            _lut_backward_kernel(lut.S, lut.trees, self._j, self._r_min, self._u_min,
                                  self.output, x_grad, lut.a_arr, lut.b_arr,
                                  lut.y_dim, args.context_size, args.n_t,
                                  np.float32(learning_rate))
        else:
            trees = lut.trees
            for pos in range(args.context_size):
                j_bins = self._j[pos]
                jbar_bins = j_bins ^ (1 << self._r_min[pos])
                S_j = _f32(lut.S[trees, j_bins])
                S_jbar = _f32(lut.S[trees, jbar_bins])
                gi = ((S_jbar - S_j) * self.output[pos, :lut.y_dim]).sum(axis=1)
                sign_u = np.where(self._u_min[pos] > 0, 1.0, -1.0)
                v = gi * (-0.5 * sign_u / (1 + np.abs(self._u_min[pos]))**2)
                n = x_grad.shape[1]
                idx_a = lut.a_arr[trees, self._r_min[pos]]
                idx_b = lut.b_arr[trees, self._r_min[pos]]
                x_grad[pos] += np.bincount(idx_a, weights=v, minlength=n)
                x_grad[pos] -= np.bincount(idx_b, weights=v, minlength=n)
                lut.S[trees, j_bins] -= learning_rate * self.output[pos, :lut.y_dim]

    def compute_gradients(self, args):
        VOCAB_SIZE = args.vocab_size
        for pos in range(args.context_size):
            softmax(self.output[pos], VOCAB_SIZE, 1.0)
            self.output[pos][self.tokens[pos + 1]] -= 1.0

    def sample_token(self, args):
        VOCAB_SIZE = args.vocab_size
        softmax(self.output[args.context_size - 1], VOCAB_SIZE, args.temperature)
        return sample(self.output[args.context_size - 1], VOCAB_SIZE)

    def validation_loss(self, args):
        last = args.context_size - 1
        softmax(self.output[last], args.vocab_size, 1.0)
        target = self.tokens[args.context_size]
        prob = max(self.output[last][target], 1e-30)
        return -math.log(prob)


class FactoredOutputHead:
    __slots__ = ['unembedder_hi', 'output_hi',
                 'unembedder_lo', 'output_lo',
                 'tokens_hi', 'tokens_lo',
                 '_hi_j', '_hi_r_min', '_hi_u_min',
                 '_lo_j', '_lo_r_min', '_lo_u_min']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        VOCAB_HI = args.vocab_hi
        self.unembedder_hi = LUT(N_T)
        self.output_hi = np.zeros((CS, VOCAB_HI), dtype=np.float32)
        self.unembedder_lo = LUT(N_T)
        self.output_lo = np.zeros((CS, 256), dtype=np.float32)
        self.tokens_hi = np.zeros(CS + 1, dtype=np.int32)
        self.tokens_lo = np.zeros(CS + 1, dtype=np.int32)
        self._hi_j = np.zeros((CS, N_T), dtype=np.int32)
        self._hi_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._hi_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._lo_j = np.zeros((CS, N_T), dtype=np.int32)
        self._lo_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._lo_u_min = np.zeros((CS, N_T), dtype=np.float32)

    def build(self, n_c, args):
        build_LUT(self.unembedder_hi, n_c, args.vocab_hi, args)
        build_LUT(self.unembedder_lo, n_c, 256, args)

    def load_targets(self, tokens, context_size):
        t = tokens[:context_size + 1]
        self.tokens_hi[:context_size + 1] = t // 256
        self.tokens_lo[:context_size + 1] = t % 256

    def forward(self, z, args):
        self.output_hi[:] = 0.0
        self.output_lo[:] = 0.0
        cache_index_batch(self.unembedder_hi, z, self._hi_j, self._hi_r_min, self._hi_u_min)
        cache_index_batch(self.unembedder_lo, z, self._lo_j, self._lo_r_min, self._lo_u_min)
        self.output_hi[:, :self.unembedder_hi.y_dim] += _f32(self.unembedder_hi.S[self.unembedder_hi.trees, self._hi_j]).sum(axis=1)
        self.output_lo[:, :self.unembedder_lo.y_dim] += _f32(self.unembedder_lo.S[self.unembedder_lo.trees, self._lo_j]).sum(axis=1)

    def backward(self, x_grad, args):
        global learning_rate
        for lut, j, r_min, u_min, output in [
            (self.unembedder_hi, self._hi_j, self._hi_r_min, self._hi_u_min, self.output_hi),
            (self.unembedder_lo, self._lo_j, self._lo_r_min, self._lo_u_min, self.output_lo),
        ]:
            if HAS_NUMBA and lut.S.dtype == np.float32:
                _lut_backward_kernel(lut.S, lut.trees, j, r_min, u_min,
                                      output, x_grad, lut.a_arr, lut.b_arr,
                                      lut.y_dim, args.context_size, args.n_t,
                                      np.float32(learning_rate))
            else:
                trees = lut.trees
                for pos in range(args.context_size):
                    j_bins = j[pos]
                    jbar_bins = j_bins ^ (1 << r_min[pos])
                    S_j = _f32(lut.S[trees, j_bins])
                    S_jbar = _f32(lut.S[trees, jbar_bins])
                    gi = ((S_jbar - S_j) * output[pos, :lut.y_dim]).sum(axis=1)
                    sign_u = np.where(u_min[pos] > 0, 1.0, -1.0)
                    v = gi * (-0.5 * sign_u / (1 + np.abs(u_min[pos]))**2)
                    n = x_grad.shape[1]
                    idx_a = lut.a_arr[trees, r_min[pos]]
                    idx_b = lut.b_arr[trees, r_min[pos]]
                    x_grad[pos] += np.bincount(idx_a, weights=v, minlength=n)
                    x_grad[pos] -= np.bincount(idx_b, weights=v, minlength=n)
                    lut.S[trees, j_bins] -= learning_rate * output[pos, :lut.y_dim]

    def compute_gradients(self, args):
        VOCAB_HI = args.vocab_hi
        for pos in range(args.context_size):
            softmax(self.output_hi[pos], VOCAB_HI, 1.0)
            self.output_hi[pos][self.tokens_hi[pos + 1]] -= 1.0
            softmax(self.output_lo[pos], 256, 1.0)
            self.output_lo[pos][self.tokens_lo[pos + 1]] -= 1.0

    def sample_token(self, args):
        VOCAB_SIZE = args.vocab_size
        VOCAB_HI = args.vocab_hi
        softmax(self.output_hi[args.context_size - 1], VOCAB_HI, args.temperature)
        hi = sample(self.output_hi[args.context_size - 1], VOCAB_HI)
        softmax(self.output_lo[args.context_size - 1], 256, args.temperature)
        lo = sample(self.output_lo[args.context_size - 1], 256)
        token_id = hi * 256 + lo
        if token_id >= VOCAB_SIZE:
            token_id = hi * 256
        return token_id

    def validation_loss(self, args):
        last = args.context_size - 1
        softmax(self.output_hi[last], args.vocab_hi, 1.0)
        softmax(self.output_lo[last], 256, 1.0)
        target_hi = self.tokens_hi[args.context_size]
        target_lo = self.tokens_lo[args.context_size]
        prob_hi = max(self.output_hi[last][target_hi], 1e-30)
        prob_lo = max(self.output_lo[last][target_lo], 1e-30)
        return -math.log(prob_hi) + -math.log(prob_lo)


class Model:
    __slots__ = ['Token_embedder', 'tokens', 'z', '_x_buf', 'FFN',
                 '_ffn_j', '_ffn_r_min', '_ffn_u_min', 'head', 'output_head']
    def __init__(self, args):
        VOCAB_SIZE = args.vocab_size
        EMBEDDING_DIM = args.embedding_dim
        NUM_LAYERS = args.num_layers
        NUM_HEADS = args.num_heads
        N_T = args.n_t
        CS = args.context_size

        self.Token_embedder = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
        self.tokens = np.zeros(CS + 1, dtype=np.int32)
        self.z = np.zeros((CS, EMBEDDING_DIM), dtype=np.float32)
        self._x_buf = np.zeros_like(self.z)

        self.FFN = [LUT(N_T) for _ in range(NUM_LAYERS)]
        self._ffn_j = [np.zeros((CS, N_T), dtype=np.int32) for _ in range(NUM_LAYERS)]
        self._ffn_r_min = [np.zeros((CS, N_T), dtype=np.int32) for _ in range(NUM_LAYERS)]
        self._ffn_u_min = [np.zeros((CS, N_T), dtype=np.float32) for _ in range(NUM_LAYERS)]

        self.head = [[AttentionHead(args) for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]

        if args.factored_output:
            self.output_head = FactoredOutputHead(args)
        else:
            self.output_head = StandardOutputHead(args)


class TrainingData:
    __slots__ = ['data', 'length', 'val_data', 'val_length', 'testing_input_data']
    def __init__(self):
        self.data = None
        self.length = 0
        self.val_data = None
        self.val_length = 0
        self.testing_input_data = None


# ---------------------------------------------------------------------------
# LUT operations
# ---------------------------------------------------------------------------

def build_LUT(lut, total_n_c, y_dim, args):
    N_T = args.n_t
    N_C = args.n_c
    EMBEDDING_DIM = args.embedding_dim
    s_dtype = np.float16 if getattr(args, 'fp16', False) else np.float32

    lut.y_dim = y_dim
    num_bins = 1 << total_n_c
    lut.S = np.zeros((N_T, num_bins, y_dim), dtype=s_dtype)
    lut.a_arr = np.zeros((N_T, N_C), dtype=np.int32)
    lut.b_arr = np.zeros((N_T, N_C), dtype=np.int32)
    lut.bitmasks = (1 << np.arange(N_C)).astype(np.int32)
    for i in range(N_T):
        lut.a_arr[i] = fill_vector_with_random_integers(N_C, EMBEDDING_DIM)
        lut.b_arr[i] = fill_vector_with_random_integers_different_from_vector2(
            lut.a_arr[i], N_C, EMBEDDING_DIM)


def cache_index_batch(lut, x_batch, j_out, r_min_out, u_min_out):
    """Batch cache_index across all positions. x_batch: (CS, dim), outputs: (CS, N_T)."""
    u = x_batch[:, lut.a_arr] - x_batch[:, lut.b_arr]      # (CS, N_T, N_C)
    j_out[:] = ((u > 0).astype(np.int32) * lut.bitmasks).sum(axis=2)
    abs_u = np.abs(u)
    r_min_out[:] = abs_u.argmin(axis=2)
    cs_idx = np.arange(x_batch.shape[0])[:, np.newaxis]
    u_min_out[:] = u[cs_idx, lut.trees, r_min_out]


def cache_PE_index_batch(u_batch, j_out, r_min_out, u_min_out, args):
    """Batch cache_PE_index across all positions. u_batch: (CS, N_T, PD), outputs: (CS, N_T)."""
    j_out[:] = ((u_batch > 0).astype(np.int32) * args.pe_bitmasks).sum(axis=2)
    abs_u = np.abs(u_batch)
    r_min_out[:] = abs_u.argmin(axis=2)
    cs_idx = np.arange(u_batch.shape[0])[:, np.newaxis]
    tree_idx = np.arange(u_batch.shape[1])
    u_min_out[:] = u_batch[cs_idx, tree_idx, r_min_out]


def _f32(arr):
    """Cast to float32 if needed (no-op if already float32)."""
    return arr if arr.dtype == np.float32 else arr.astype(np.float32)


def CONCATENATE_vec(Q, P, PE, args):
    """Vectorized CONCATENATE for arrays of indices."""
    return (Q << args.shift_qk) | (P << args.positional_dim) | PE


# ---------------------------------------------------------------------------
# Numba JIT kernels (fused loops — no temporary allocations)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _attention_forward_kernel(S, all_jV, all_jPE, y, shift_qk, pd, CS, N_T, y_dim):
    """Fused attention forward: eliminates all (P, N_T, y_dim) temporaries."""
    for pos in prange(1, CS):
        for pos1 in range(pos):
            for t in range(N_T):
                j = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pos - pos1, t]
                for d in range(y_dim):
                    y[pos, d] += S[t, j, d]


@njit(parallel=True, cache=True)
def _attention_backward_kernel(S, all_jV, all_rV, all_uV, all_jPE, all_rPE, all_uPE,
                                y_grad, x_grad, pos_grad, a_arr, b_arr,
                                shift_qk, pd, CS, N_T, y_dim, lr):
    """Fused attention backward with parallel gradient computation.

    Three-phase approach using per-thread buffers to avoid O(CS^2) memory:
      Phase 1 (parallel over pos): each thread accumulates x_grad and pe_grad
        contributions into its own buffer. x_grad[pos] (q_wins) goes directly;
        x_grad[pos1] (k_wins) goes into per-thread buffer.
      Phase 2: reduce per-thread buffers into x_grad and pos_grad.
      Phase 3 (sequential): update S tables.
    """
    ED = x_grad.shape[1]
    n_threads = get_num_threads()
    # Per-thread buffers for contributions that cross pos boundaries
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)
    pe_grad_thr = np.zeros((n_threads, CS, N_T, pd), dtype=np.float32)

    # Phase 1: parallel gradient computation
    for pos in prange(1, CS):
        tid = get_thread_id()
        for pos1 in range(pos):
            pe_pos = pos - pos1
            for t in range(N_T):
                jQ = all_jV[pos, t]
                jK = all_jV[pos1, t]
                j_bin = (jQ << shift_qk) | (jK << pd) | all_jPE[pe_pos, t]

                abs_uQ = abs(all_uV[pos, t])
                abs_uK = abs(all_uV[pos1, t])
                q_wins = abs_uQ < abs_uK

                if q_wins:
                    flip_bit = all_rV[pos, t] + shift_qk
                else:
                    flip_bit = all_rV[pos1, t] + pd
                jbar = j_bin ^ (1 << flip_bit)

                gi = 0.0
                for d in range(y_dim):
                    gi += (S[t, jbar, d] - S[t, j_bin, d]) * y_grad[pos, d]

                if q_wins:
                    u_min = all_uV[pos, t]
                    r = all_rV[pos, t]
                else:
                    u_min = all_uV[pos1, t]
                    r = all_rV[pos1, t]

                sign_u = 1.0 if u_min > 0.0 else -1.0
                abs_u_min = abs(u_min)
                v = gi * (-0.5 * sign_u / (1.0 + abs_u_min) ** 2)

                idx_a = a_arr[t, r]
                idx_b = b_arr[t, r]
                if q_wins:
                    # x_grad[pos] is unique per thread's pos — safe to write directly
                    x_grad[pos, idx_a] += v
                    x_grad[pos, idx_b] -= v
                else:
                    # x_grad[pos1] may conflict — accumulate in per-thread buffer
                    x_grad_thr[tid, pos1, idx_a] += v
                    x_grad_thr[tid, pos1, idx_b] -= v

                # PE branch
                abs_uPE = abs(all_uPE[pe_pos, t])
                if abs_uPE < abs_uQ and abs_uPE < abs_uK:
                    rPE = all_rPE[pe_pos, t]
                    jbarPE = j_bin ^ (1 << rPE)
                    giPE = 0.0
                    for d in range(y_dim):
                        giPE += (S[t, jbarPE, d] - S[t, j_bin, d]) * y_grad[pos, d]
                    u_pe = all_uPE[pe_pos, t]
                    sign_pe = 1.0 if u_pe > 0.0 else -1.0
                    deltaPE = giPE * (-0.5 * sign_pe / (1.0 + abs(u_pe)) ** 2)
                    pe_grad_thr[tid, pe_pos, t, rPE] += deltaPE

    # Phase 2: reduce per-thread buffers
    for target in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, target, d]
            x_grad[target, d] += total

    for pe_pos in prange(CS):
        for t in range(N_T):
            for r in range(pd):
                total = np.float32(0.0)
                for thr in range(n_threads):
                    total += pe_grad_thr[thr, pe_pos, t, r]
                pos_grad[pe_pos, t, r] += total

    # Phase 3: S update (sequential — different pos can hit same S entry)
    for pos in range(1, CS):
        for pos1 in range(pos):
            for t in range(N_T):
                j_bin = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pos - pos1, t]
                for d in range(y_dim):
                    S[t, j_bin, d] -= lr * y_grad[pos, d]


@njit(cache=True)
def _lut_backward_kernel(S, trees, j, r_min, u_min, grad, x_grad, a_arr, b_arr,
                          y_dim, CS, N_T, lr):
    """Fused backward for simple LUT (FFN, output head). grad: (CS, >=y_dim)."""
    for pos in range(CS):
        for t in range(N_T):
            tree = trees[t]
            j_bin = j[pos, t]
            r = r_min[pos, t]
            jbar_bin = j_bin ^ (1 << r)

            gi = 0.0
            for d in range(y_dim):
                gi += (S[tree, jbar_bin, d] - S[tree, j_bin, d]) * grad[pos, d]

            u = u_min[pos, t]
            sign_u = 1.0 if u > 0.0 else -1.0
            abs_u = abs(u)
            v = gi * (-0.5 * sign_u / (1.0 + abs_u) ** 2)

            idx_a = a_arr[t, r]
            idx_b = b_arr[t, r]
            x_grad[pos, idx_a] += v
            x_grad[pos, idx_b] -= v

            for d in range(y_dim):
                S[tree, j_bin, d] -= lr * grad[pos, d]


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------

def build_Model(m, args):
    VOCAB_SIZE = args.vocab_size
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    
    N_T = args.n_t

    m.Token_embedder[:] = random_vector(VOCAB_SIZE * EMBEDDING_DIM, 1.0).reshape(VOCAB_SIZE, EMBEDDING_DIM)

    for l in range(NUM_LAYERS):
        build_LUT(m.FFN[l], N_C, EMBEDDING_DIM, args)
        for h in range(NUM_HEADS):
            m.head[l][h].Positional_encoding[:] = random_vector(
                args.context_size * N_T * POSITIONAL_DIM, 1.0
            ).reshape(args.context_size, N_T, POSITIONAL_DIM)
            build_LUT(m.head[l][h].V, N_C + N_C + POSITIONAL_DIM, EMBEDDING_DIM, args)

    m.output_head.build(N_C, args)


def attention_forward(head, x, y, args):
    CS = args.context_size
    lut = head.V
    trees = lut.trees
    y_dim = lut.y_dim

    cache_index_batch(lut, x, head._v_j, head._v_r_min, head._v_u_min)
    cache_PE_index_batch(head.Positional_encoding, head._pe_j, head._pe_r_min, head._pe_u_min, args)

    all_jV = head._v_j    # (CS, N_T) — zero-copy
    all_jPE = head._pe_j  # (CS, N_T) — zero-copy

    if HAS_NUMBA and lut.S.dtype == np.float32:
        _attention_forward_kernel(lut.S, all_jV, all_jPE, y,
                                   args.shift_qk, args.positional_dim,
                                   CS, args.n_t, y_dim)
    else:
        for pos in range(1, CS):
            P = pos
            jQ = all_jV[pos]
            jK = all_jV[:P]
            jPE = all_jPE[pos - np.arange(P)]
            j_bins = CONCATENATE_vec(jQ, jK, jPE, args)
            y[pos, :y_dim] += _f32(lut.S[trees, j_bins]).sum(axis=1).sum(axis=0)


def attention_backward(head, x_grad, y_grad, args):
    global learning_rate
    CS = args.context_size
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim
    lut = head.V
    trees = lut.trees
    y_dim = lut.y_dim

    pos_grad = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

    # Cache arrays already computed in attention_forward — zero-copy references
    all_jV = head._v_j        # (CS, N_T)
    all_rV = head._v_r_min    # (CS, N_T)
    all_uV = head._v_u_min    # (CS, N_T)
    all_jPE = head._pe_j      # (CS, N_T)
    all_rPE = head._pe_r_min  # (CS, N_T)
    all_uPE = head._pe_u_min  # (CS, N_T)

    if HAS_NUMBA and lut.S.dtype == np.float32:
        _attention_backward_kernel(lut.S, all_jV, all_rV, all_uV,
                                    all_jPE, all_rPE, all_uPE,
                                    y_grad, x_grad, pos_grad,
                                    lut.a_arr, lut.b_arr,
                                    args.shift_qk, POSITIONAL_DIM,
                                    CS, N_T, y_dim,
                                    np.float32(learning_rate))
    else:
        ED = x_grad.shape[1]
        flat_size = x_grad.size
        abs_all_uV = np.abs(all_uV)
        abs_all_uPE = np.abs(all_uPE)
        trees_b_full = np.broadcast_to(trees, (CS, N_T))
        pos1_grid = np.broadcast_to(np.arange(CS)[:, np.newaxis], (CS, N_T))

        for pos in range(1, CS):
            P = pos
            y_g = y_grad[pos, :y_dim]

            jQ = all_jV[pos]
            rQ = all_rV[pos]
            abs_uQ = abs_all_uV[pos]

            jK = all_jV[:P]
            rK = all_rV[:P]
            abs_uK = abs_all_uV[:P]

            pe_idx = pos - np.arange(P)
            jPE = all_jPE[pe_idx]
            rPE = all_rPE[pe_idx]

            j_bins = CONCATENATE_vec(jQ, jK, jPE, args)
            S_j = _f32(lut.S[trees, j_bins])

            q_mask = abs_uQ < abs_uK
            flip_bit = np.where(q_mask, rQ + args.shift_qk, rK + POSITIONAL_DIM)
            jbar_bins = j_bins ^ (1 << flip_bit)
            S_jbar = _f32(lut.S[trees, jbar_bins])

            gi = ((S_jbar - S_j) * y_g).sum(axis=2)
            uQ_vals = all_uV[pos]
            uK_vals = all_uV[:P]
            u_min_qk = np.where(q_mask, uQ_vals, uK_vals)
            abs_u_min_qk = np.where(q_mask, abs_uQ, abs_uK)
            sign_u = np.where(u_min_qk > 0, 1.0, -1.0)
            v = gi * (-0.5 * sign_u / (1 + abs_u_min_qk)**2)
            r_min_qk = np.where(q_mask, rQ, rK)

            trees_b = trees_b_full[:P]

            if q_mask.any():
                q_t = trees_b[q_mask]
                q_r = r_min_qk[q_mask]
                q_v = v[q_mask]
                x_grad[pos] += np.bincount(lut.a_arr[q_t, q_r], weights=q_v, minlength=ED)
                x_grad[pos] -= np.bincount(lut.b_arr[q_t, q_r], weights=q_v, minlength=ED)

            k_mask = ~q_mask
            if k_mask.any():
                k_t = trees_b[k_mask]
                k_r = r_min_qk[k_mask]
                k_v = v[k_mask]
                k_pos1 = pos1_grid[:P][k_mask]
                flat_a = k_pos1 * ED + lut.a_arr[k_t, k_r]
                flat_b = k_pos1 * ED + lut.b_arr[k_t, k_r]
                x_grad.ravel()[:] += np.bincount(flat_a, weights=k_v, minlength=flat_size)
                x_grad.ravel()[:] -= np.bincount(flat_b, weights=k_v, minlength=flat_size)

            abs_uPE = abs_all_uPE[pe_idx]
            pe_mask = (abs_uPE < abs_uQ) & (abs_uPE < abs_uK)
            if pe_mask.any():
                jbarPE_bins = j_bins ^ (1 << rPE)
                S_jbarPE = _f32(lut.S[trees, jbarPE_bins])
                giPE = ((S_jbarPE - S_j) * y_g).sum(axis=2)
                uPE_vals = all_uPE[pe_idx]
                u_pe = uPE_vals[pe_mask]
                sign_pe = np.where(u_pe > 0, 1.0, -1.0)
                deltaPE = giPE[pe_mask] * (-0.5 * sign_pe / (1 + np.abs(u_pe))**2)
                pe_t = trees_b[pe_mask]
                pe_r = rPE[pe_mask]
                pe_pair = pos1_grid[:P][pe_mask]
                np.add.at(pos_grad, (pe_idx[pe_pair], pe_t, pe_r), deltaPE)

            np.subtract.at(lut.S, (trees_b.ravel(), j_bins.ravel()),
                            np.float32(learning_rate) * y_g)

    head.Positional_encoding -= learning_rate * pos_grad


def model_forward(m, args):
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    for l in range(NUM_LAYERS):
        np.copyto(m._x_buf, m.z)
        for h in range(NUM_HEADS):
            attention_forward(m.head[l][h], m._x_buf, m.z, args)

        cache_index_batch(m.FFN[l], m.z, m._ffn_j[l], m._ffn_r_min[l], m._ffn_u_min[l])
        lut = m.FFN[l]
        m.z[:, :lut.y_dim] += _f32(lut.S[lut.trees, m._ffn_j[l]]).sum(axis=1)

    m.output_head.forward(m.z, args)


def model_backward(m, args):
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    x_grad = np.zeros((args.context_size, EMBEDDING_DIM), dtype=np.float32)

    m.output_head.backward(x_grad, args)

    for l in range(NUM_LAYERS - 1, -1, -1):
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        lut = m.FFN[l]
        trees = lut.trees
        CS = args.context_size
        y_dim = lut.y_dim
        j = m._ffn_j[l]
        r_min = m._ffn_r_min[l]
        u_min = m._ffn_u_min[l]

        # FFN backward uses batch-read semantics: read S for ALL positions
        # before any updates. This can't be fused into a per-position kernel
        # because positions share bins (2^n_c is small).
        jbar = j ^ (1 << r_min)
        S_j = _f32(lut.S[trees, j])
        S_jbar = _f32(lut.S[trees, jbar])
        gi = ((S_jbar - S_j) * y_grad[:, np.newaxis, :y_dim]).sum(axis=2)
        sign_u = np.where(u_min > 0, 1.0, -1.0)
        v = gi * (-0.5 * sign_u / (1 + np.abs(u_min))**2)

        ED = EMBEDDING_DIM
        pos_idx = np.arange(CS)[:, np.newaxis]
        idx_a = lut.a_arr[trees, r_min]
        idx_b = lut.b_arr[trees, r_min]
        flat_a = (pos_idx * ED + idx_a).ravel()
        flat_b = (pos_idx * ED + idx_b).ravel()
        v_flat = v.ravel()
        flat_size = x_grad.size
        x_grad.ravel()[:] += np.bincount(flat_a, weights=v_flat, minlength=flat_size)
        x_grad.ravel()[:] -= np.bincount(flat_b, weights=v_flat, minlength=flat_size)

        for pos in range(CS):
            lut.S[trees, j[pos]] -= learning_rate * y_grad[pos, :y_dim]

        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        for h in range(NUM_HEADS):
            attention_backward(m.head[l][h], x_grad, y_grad, args)

    # no need to compute gradients for the embedder; just update the synaptic values
    # (disabled in the C version too)
    # for pos in range(args.context_size):
    #     for k in range(EMBEDDING_DIM):
    #         m.Token_embedder[m.tokens[pos]][k] -= learning_rate * x_grad[pos][k]


def model_training_step(m, args):
    model_forward(m, args)
    m.output_head.compute_gradients(args)
    model_backward(m, args)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_training_data(training, args):
    
    TESTING_LENGTH = args.testing_length
    enc = args.enc

    # Load and tokenize training data
    try:
        with open(args.training_data, 'r', encoding='utf-8') as f:
            text = f.read()
    except IOError:
        print(f"Error opening training datafile {args.training_data}")
        sys.exit(1)

    tokens = enc.encode(text)
    training.data = np.array(tokens, dtype=np.int32)
    training.length = len(training.data) - args.context_size - 1
    print(f"Training data: {len(training.data)} tokens from {args.training_data}")

    # Load or split validation data
    if args.validation_data:
        try:
            with open(args.validation_data, 'r', encoding='utf-8') as f:
                val_text = f.read()
        except IOError:
            print(f"Error opening validation datafile {args.validation_data}")
            sys.exit(1)
        val_tokens = enc.encode(val_text)
        training.val_data = np.array(val_tokens, dtype=np.int32)
        training.val_length = len(training.val_data) - args.context_size - 1
        print(f"Validation data: {len(training.val_data)} tokens from {args.validation_data}")
    else:
        # No separate validation file: use random snippets from training data
        training.val_data = training.data
        training.val_length = training.length
        print("Validation data: sampled from training data")

    # Sample validation snippet indices from val data
    training.testing_input_data = np.zeros(TESTING_LENGTH, dtype=np.int32)
    for i in range(TESTING_LENGTH):
        training.testing_input_data[i] = random.randint(0, training.val_length - 1)

    print(f"Tokenizer: {args.tokenizer}, vocab_size: {args.vocab_size}")


def get_random_training_index(training):
    return random.randint(0, training.length - 1)


def load_snippet(m, data_array, char_start, args):
    CS = args.context_size
    m.tokens[:CS + 1] = data_array[char_start:char_start + CS + 1]
    m.z[:] = m.Token_embedder[m.tokens[:CS]]
    m.output_head.load_targets(m.tokens, CS)


def model_inference(m, args):
    model_forward(m, args)
    return m.output_head.sample_token(args)


def model_prompt_response(m, prompt_text, response_length, args):
    enc = args.enc

    # Encode prompt to token IDs
    prompt_tokens = enc.encode(prompt_text)
    # Truncate or pad to args.context_size
    if len(prompt_tokens) > args.context_size:
        prompt_tokens = prompt_tokens[:args.context_size]
    else:
        prompt_tokens = [0] * (args.context_size - len(prompt_tokens)) + list(prompt_tokens)

    # Print the prompt
    sys.stdout.write(prompt_text[:80])

    for i in range(response_length):
        m.z[:] = m.Token_embedder[prompt_tokens]
        response = model_inference(m, args)
        sys.stdout.write(enc.decode([response]))

        # Shift context window and append new token
        prompt_tokens = prompt_tokens[1:] + [response]

    sys.stdout.flush()


def print_model_stats(m, args):
    """Print model parameter count and memory footprint."""
    total_params = 0
    total_bytes = 0

    def count_array(arr):
        nonlocal total_params, total_bytes
        total_params += arr.size
        total_bytes += arr.nbytes

    # Token embedder
    count_array(m.Token_embedder)

    # Per-layer
    for l in range(args.num_layers):
        # FFN LUT
        count_array(m.FFN[l].S)
        # Attention heads
        for h in range(args.num_heads):
            head = m.head[l][h]
            count_array(head.V.S)
            count_array(head.Positional_encoding)

    # Output head
    if args.factored_output:
        count_array(m.output_head.unembedder_hi.S)
        count_array(m.output_head.unembedder_lo.S)
    else:
        count_array(m.output_head.unembedder.S)

    if total_bytes < 1024**2:
        mem_str = f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024**3:
        mem_str = f"{total_bytes / 1024**2:.1f} MB"
    else:
        mem_str = f"{total_bytes / 1024**3:.2f} GB"

    if total_params < 1_000_000:
        param_str = f"{total_params:,}"
    else:
        param_str = f"{total_params / 1_000_000:.1f}M"

    dtype_str = "fp16" if getattr(args, 'fp16', False) else "fp32"
    print(f"Model: {param_str} parameters, {mem_str} memory ({dtype_str})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global learning_rate

    parser = argparse.ArgumentParser(description="SNN Transformer (Python port)")
    parser.add_argument('training_data', type=str, help='Path to training data file')
    parser.add_argument('--validation-data', type=str, default=None,
                        help='Path to validation data file (if not provided, samples from training data)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='tiktoken encoding name (default: gpt2)')
    parser.add_argument('--context-size', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='(auto-detected from tokenizer, this default is overridden)')
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--positional-dim', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--n-t', type=int, default=16)
    parser.add_argument('--n-c', type=int, default=6)
    parser.add_argument('--testing-length', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=100000000)
    parser.add_argument('--validation-interval', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--loss-file', type=str, default='loss.csv')
    parser.add_argument('--factored-output', action='store_true',
                        help='Use factored base-256 decomposition for unembedder (faster for large vocabs)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use float16 for LUT S tables (halves memory, may reduce precision)')
    args = parser.parse_args()

    # Initialize tokenizer and auto-set vocab size
    enc = tiktoken.get_encoding(args.tokenizer)
    args.vocab_size = enc.n_vocab
    args.enc = enc

    # Factored output dimensions
    args.vocab_hi = (args.vocab_size + 255) // 256
    args.vocab_lo = 256

    # Pre-compute positional encoding bitmasks and shift constants
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim

    # Initialize loss file with header
    with open(args.loss_file, 'w') as f:
        f.write("step, loss, perplexity\n")

    training = TrainingData()
    load_training_data(training, args)

    m = Model(args)
    build_Model(m, args)
    print_model_stats(m, args)

    pbar = tqdm(range(args.max_steps), desc="Training", unit="step")
    last_ppl = None
    last_loss = None

    for t in pbar:
        load_snippet(m, training.data, get_random_training_index(training), args)

        # Adam learning rate scheduler
        learning_rate = min(1.0 / math.sqrt(1 + t), t / 4000.0 / math.sqrt(4000))

        model_training_step(m, args)

        if t % args.validation_interval == 0:
            pbar.set_description("Validating")

            validation_loss = 0.0
            for i in tqdm(range(args.testing_length), desc="  Val", leave=False, unit="snip"):
                load_snippet(m, training.val_data, int(training.testing_input_data[i]), args)
                model_forward(m, args)
                validation_loss += m.output_head.validation_loss(args)
            validation_loss /= args.testing_length

            perplexity = math.exp(validation_loss)
            last_ppl = perplexity
            last_loss = validation_loss

            with open(args.loss_file, 'a') as f:
                f.write(f"{t}, {validation_loss:.6f}, {perplexity:.2f}\n")

            # Print validation result and sample generation
            tqdm.write(f"\n--- step {t:,} | ppl={perplexity:.2f} | loss={validation_loss:.3f} ---")
            val_idx = random.randint(0, training.val_length - 1)
            prompt_tokens = training.val_data[val_idx:val_idx + args.context_size].tolist()
            prompt_text = enc.decode(prompt_tokens)
            # Capture generation output
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            model_prompt_response(m, prompt_text, 80, args)
            sys.stdout = old_stdout
            tqdm.write(buf.getvalue())
            tqdm.write("")

            pbar.set_description("Training")

        if last_ppl is not None:
            pbar.set_postfix(ppl=f"{last_ppl:.2f}", loss=f"{last_loss:.3f}", lr=f"{learning_rate:.4f}")


if __name__ == '__main__':
    main()
