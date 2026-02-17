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
    return int(np.searchsorted(cumsum, random.random()))


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


def Up(x):
    """Surrogate gradient of the Heaviside step function."""
    sign_x = 1 if x > 0 else -1  # zero has "minus" sign
    return -0.5 * sign_x / (1 + abs(x)) / (1 + abs(x))


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


class LUTcache:
    __slots__ = ['r_min', 'u_min', 'j']
    def __init__(self, n_t):
        self.r_min = np.zeros(n_t, dtype=np.int32)
        self.u_min = np.zeros(n_t, dtype=np.float32)
        self.j = np.zeros(n_t, dtype=np.int32)


class AttentionHead:
    __slots__ = ['V', 'V_cache', 'Positional_encoding', 'PE_cache']
    def __init__(self, args):
        N_T = args.n_t
        POSITIONAL_DIM = args.positional_dim
        self.V = LUT(N_T)
        self.V_cache = [LUTcache(N_T) for _ in range(args.context_size)]
        self.Positional_encoding = np.zeros((args.context_size, N_T, POSITIONAL_DIM), dtype=np.float32)
        self.PE_cache = [LUTcache(N_T) for _ in range(args.context_size)]


class StandardOutputHead:
    __slots__ = ['unembedder', 'unembedder_vars', 'output', 'tokens']
    def __init__(self, args):
        N_T = args.n_t
        VOCAB_SIZE = args.vocab_size
        self.unembedder = LUT(N_T)
        self.unembedder_vars = [LUTcache(N_T) for _ in range(args.context_size)]
        self.output = np.zeros((args.context_size, VOCAB_SIZE), dtype=np.float32)
        self.tokens = None  # set by load_targets

    def build(self, n_c, args):
        build_LUT(self.unembedder, n_c, args.vocab_size, args)

    def load_targets(self, tokens, context_size):
        self.tokens = tokens

    def forward(self, z, args):
        self.output[:] = 0.0
        for pos in range(args.context_size):
            cache_index(self.unembedder, self.unembedder_vars[pos], z[pos], args)
            LUT_forward(self.unembedder, self.unembedder_vars[pos], self.output[pos], args)

    def backward(self, x_grad, args):
        for pos in range(args.context_size):
            LUT_backward(self.unembedder, self.unembedder_vars[pos], x_grad[pos], self.output[pos], args)

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
    __slots__ = ['unembedder_hi', 'unembedder_hi_vars', 'output_hi',
                 'unembedder_lo', 'unembedder_lo_vars', 'output_lo',
                 'tokens_hi', 'tokens_lo']
    def __init__(self, args):
        N_T = args.n_t
        VOCAB_HI = args.vocab_hi
        self.unembedder_hi = LUT(N_T)
        self.unembedder_hi_vars = [LUTcache(N_T) for _ in range(args.context_size)]
        self.output_hi = np.zeros((args.context_size, VOCAB_HI), dtype=np.float32)
        self.unembedder_lo = LUT(N_T)
        self.unembedder_lo_vars = [LUTcache(N_T) for _ in range(args.context_size)]
        self.output_lo = np.zeros((args.context_size, 256), dtype=np.float32)
        self.tokens_hi = np.zeros(args.context_size + 1, dtype=np.int32)
        self.tokens_lo = np.zeros(args.context_size + 1, dtype=np.int32)

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
        for pos in range(args.context_size):
            cache_index(self.unembedder_hi, self.unembedder_hi_vars[pos], z[pos], args)
            LUT_forward(self.unembedder_hi, self.unembedder_hi_vars[pos], self.output_hi[pos], args)
            cache_index(self.unembedder_lo, self.unembedder_lo_vars[pos], z[pos], args)
            LUT_forward(self.unembedder_lo, self.unembedder_lo_vars[pos], self.output_lo[pos], args)

    def backward(self, x_grad, args):
        for pos in range(args.context_size):
            LUT_backward(self.unembedder_hi, self.unembedder_hi_vars[pos], x_grad[pos], self.output_hi[pos], args)
            LUT_backward(self.unembedder_lo, self.unembedder_lo_vars[pos], x_grad[pos], self.output_lo[pos], args)

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
    __slots__ = ['Token_embedder', 'tokens', 'z', '_x_buf', 'FFN', 'FFN_cache', 'head', 'output_head']
    def __init__(self, args):
        VOCAB_SIZE = args.vocab_size
        EMBEDDING_DIM = args.embedding_dim
        NUM_LAYERS = args.num_layers
        NUM_HEADS = args.num_heads
        N_T = args.n_t

        self.Token_embedder = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
        self.tokens = np.zeros(args.context_size + 1, dtype=np.int32)
        self.z = np.zeros((args.context_size, EMBEDDING_DIM), dtype=np.float32)
        self._x_buf = np.zeros_like(self.z)

        self.FFN = [LUT(N_T) for _ in range(NUM_LAYERS)]
        self.FFN_cache = [[LUTcache(N_T) for _ in range(args.context_size)] for _ in range(NUM_LAYERS)]

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


def cache_index(lut, cache, x, args):
    u = x[lut.a_arr] - x[lut.b_arr]                        # (N_T, N_C)
    cache.j[:] = ((u > 0).astype(np.int32) * lut.bitmasks).sum(axis=1)
    abs_u = np.abs(u)
    cache.r_min[:] = abs_u.argmin(axis=1)
    cache.u_min[:] = u[lut.trees, cache.r_min]


def cache_PE_index(cache, u, args):
    cache.j[:] = ((u > 0).astype(np.int32) * args.pe_bitmasks).sum(axis=1)
    abs_u = np.abs(u)
    cache.r_min[:] = abs_u.argmin(axis=1)
    cache.u_min[:] = u[np.arange(u.shape[0]), cache.r_min]


def _f32(arr):
    """Cast to float32 if needed (no-op if already float32)."""
    return arr if arr.dtype == np.float32 else arr.astype(np.float32)


def LUT_forward(lut, cache, y, args):
    y[:lut.y_dim] += _f32(lut.S[lut.trees, cache.j]).sum(axis=0)


def backward_update(lut, cache, i, j_bin, jbar_bin, y_gradient, gradient):
    """Equivalent of BACKWARD_UPDATE macro. Uses bin indices (not flat offsets)."""
    gi = np.dot(y_gradient[:lut.y_dim], _f32(lut.S[i][jbar_bin]) - _f32(lut.S[i][j_bin]))
    v = gi * Up(cache.u_min[i])
    gradient[lut.a_arr[i, cache.r_min[i]]] += v
    gradient[lut.b_arr[i, cache.r_min[i]]] -= v


def LUT_backward(lut, cache, x_gradient, y_gradient, args):
    global learning_rate
    trees = lut.trees
    j_bins = cache.j
    jbar_bins = j_bins ^ (1 << cache.r_min)

    S_j = _f32(lut.S[trees, j_bins])
    S_jbar = _f32(lut.S[trees, jbar_bins])

    gi = ((S_jbar - S_j) * y_gradient[:lut.y_dim]).sum(axis=1)
    sign_u = np.where(cache.u_min > 0, 1.0, -1.0)
    v = gi * (-0.5 * sign_u / (1 + np.abs(cache.u_min))**2)

    n = x_gradient.shape[0]
    idx_a = lut.a_arr[trees, cache.r_min]
    idx_b = lut.b_arr[trees, cache.r_min]
    x_gradient += np.bincount(idx_a, weights=v, minlength=n)
    x_gradient -= np.bincount(idx_b, weights=v, minlength=n)

    lut.S[trees, j_bins] -= learning_rate * y_gradient[:lut.y_dim]


def CONCATENATE(Q, P, PE, args):
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    return int((Q << (N_C + POSITIONAL_DIM)) | (P << POSITIONAL_DIM) | PE)


def CONCATENATE_vec(Q, P, PE, args):
    """Vectorized CONCATENATE for arrays of indices."""
    return (Q << args.shift_qk) | (P << args.positional_dim) | PE


def concatenated_LUT_forward(lut, cacheQ, cacheK, cachePE, y, args):
    j_bins = CONCATENATE_vec(cacheQ.j, cacheK.j, cachePE.j, args)
    y[:lut.y_dim] += _f32(lut.S[lut.trees, j_bins]).sum(axis=0)


def concatenated_LUT_backward(lut, cacheQ, cacheK, cachePE,
                                x_gradientQ, x_gradientK, PE_grad, y_gradient, args):
    global learning_rate
    trees = lut.trees
    y_g = y_gradient[:lut.y_dim]

    jQ = cacheQ.j
    jK = cacheK.j
    jPE = cachePE.j
    j_bins = CONCATENATE_vec(jQ, jK, jPE, args)

    S_j = _f32(lut.S[trees, j_bins])

    # --- Q/K branch: which trees have |u_Q| < |u_K|? ---
    abs_uQ = np.abs(cacheQ.u_min)
    abs_uK = np.abs(cacheK.u_min)
    q_mask = abs_uQ < abs_uK

    jbar_Q = CONCATENATE_vec(jQ ^ (1 << cacheQ.r_min), jK, jPE, args)
    jbar_K = CONCATENATE_vec(jQ, jK ^ (1 << cacheK.r_min), jPE, args)

    jbar_bins = np.where(q_mask, jbar_Q, jbar_K)
    S_jbar = _f32(lut.S[trees, jbar_bins])

    gi = ((S_jbar - S_j) * y_g).sum(axis=1)
    u_min_qk = np.where(q_mask, cacheQ.u_min, cacheK.u_min)
    sign_u = np.where(u_min_qk > 0, 1.0, -1.0)
    v = gi * (-0.5 * sign_u / (1 + np.abs(u_min_qk))**2)

    r_min_qk = np.where(q_mask, cacheQ.r_min, cacheK.r_min)

    nQ = x_gradientQ.shape[0]
    q_trees = trees[q_mask]
    if len(q_trees) > 0:
        q_r = r_min_qk[q_mask]
        q_v = v[q_mask]
        x_gradientQ += np.bincount(lut.a_arr[q_trees, q_r], weights=q_v, minlength=nQ)
        x_gradientQ -= np.bincount(lut.b_arr[q_trees, q_r], weights=q_v, minlength=nQ)

    k_trees = trees[~q_mask]
    if len(k_trees) > 0:
        k_r = r_min_qk[~q_mask]
        k_v = v[~q_mask]
        x_gradientK += np.bincount(lut.a_arr[k_trees, k_r], weights=k_v, minlength=nQ)
        x_gradientK -= np.bincount(lut.b_arr[k_trees, k_r], weights=k_v, minlength=nQ)

    # --- PE branch ---
    abs_uPE = np.abs(cachePE.u_min)
    pe_mask = (abs_uPE < abs_uQ) & (abs_uPE < abs_uK)
    pe_trees = trees[pe_mask]
    if len(pe_trees) > 0:
        jbarPE_bins = CONCATENATE_vec(jQ, jK, jPE ^ (1 << cachePE.r_min), args)
        S_jbarPE = _f32(lut.S[pe_trees, jbarPE_bins[pe_mask]])
        giPE = ((S_jbarPE - S_j[pe_mask]) * y_g).sum(axis=1)
        u_pe = cachePE.u_min[pe_mask]
        sign_pe = np.where(u_pe > 0, 1.0, -1.0)
        deltaPE = giPE * (-0.5 * sign_pe / (1 + np.abs(u_pe))**2)
        pe_r = cachePE.r_min[pe_mask]
        np.add.at(PE_grad, (pe_trees, pe_r), deltaPE)

    # SGD update
    lut.S[trees, j_bins] -= learning_rate * y_g


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

    for pos in range(CS):
        cache_index(lut, head.V_cache[pos], x[pos], args)
        cache_PE_index(head.PE_cache[pos], head.Positional_encoding[pos], args)

    # Pre-gather all cache j values
    all_jV = np.array([head.V_cache[p].j for p in range(CS)])    # (CS, N_T)
    all_jPE = np.array([head.PE_cache[p].j for p in range(CS)])  # (CS, N_T)

    # All (pos1, pos) pairs where pos1 < pos
    all_pos1, all_pos = np.triu_indices(CS, k=1)
    jQ = all_jV[all_pos]                          # (num_pairs, N_T)
    jK = all_jV[all_pos1]                         # (num_pairs, N_T)
    jPE = all_jPE[all_pos - all_pos1]             # (num_pairs, N_T)

    j_bins = CONCATENATE_vec(jQ, jK, jPE, args)   # (num_pairs, N_T)
    S_sums = _f32(lut.S[trees, j_bins]).sum(axis=1)  # (num_pairs, y_dim)

    # Scatter-add to y by target position
    np.add.at(y[:, :y_dim], all_pos, S_sums)


def attention_backward(head, x_grad, y_grad, args):
    global learning_rate
    CS = args.context_size
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim
    lut = head.V
    trees = lut.trees
    y_dim = lut.y_dim

    pos_grad = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

    # Pre-gather all cache values into contiguous arrays
    all_jV = np.array([head.V_cache[p].j for p in range(CS)])        # (CS, N_T)
    all_rV = np.array([head.V_cache[p].r_min for p in range(CS)])    # (CS, N_T)
    all_uV = np.array([head.V_cache[p].u_min for p in range(CS)])    # (CS, N_T)
    all_jPE = np.array([head.PE_cache[p].j for p in range(CS)])      # (CS, N_T)
    all_rPE = np.array([head.PE_cache[p].r_min for p in range(CS)])  # (CS, N_T)
    all_uPE = np.array([head.PE_cache[p].u_min for p in range(CS)])  # (CS, N_T)

    for pos in range(1, CS):
        P = pos  # number of pos1 values to batch
        y_g = y_grad[pos, :y_dim]  # (y_dim,)

        # Q cache (same for all pos1, broadcasts to (P, N_T))
        jQ = all_jV[pos]     # (N_T,)
        rQ = all_rV[pos]     # (N_T,)
        uQ = all_uV[pos]     # (N_T,)

        # K caches (vary by pos1)
        jK = all_jV[:P]      # (P, N_T)
        rK = all_rV[:P]      # (P, N_T)
        uK = all_uV[:P]      # (P, N_T)

        # PE caches (indexed by pos - pos1)
        pe_idx = pos - np.arange(P)   # [pos, pos-1, ..., 1]
        jPE = all_jPE[pe_idx]   # (P, N_T)
        rPE = all_rPE[pe_idx]   # (P, N_T)
        uPE = all_uPE[pe_idx]   # (P, N_T)

        # Compute concatenated bin indices
        j_bins = CONCATENATE_vec(jQ, jK, jPE, args)  # (P, N_T)
        S_j = _f32(lut.S[trees, j_bins])  # (P, N_T, y_dim)

        # Q/K branch: which (pair, tree) elements have |u_Q| < |u_K|?
        abs_uQ = np.abs(uQ)   # (N_T,) broadcasts to (P, N_T)
        abs_uK = np.abs(uK)   # (P, N_T)
        q_mask = abs_uQ < abs_uK  # (P, N_T)

        # Compute jbar for both branches, then select
        jbar_Q = CONCATENATE_vec(jQ ^ (1 << rQ), jK, jPE, args)
        jbar_K = CONCATENATE_vec(jQ, jK ^ (1 << rK), jPE, args)
        jbar_bins = np.where(q_mask, jbar_Q, jbar_K)
        S_jbar = _f32(lut.S[trees, jbar_bins])  # (P, N_T, y_dim)

        # Gradient computation
        gi = ((S_jbar - S_j) * y_g).sum(axis=2)        # (P, N_T)
        u_min_qk = np.where(q_mask, uQ, uK)            # (P, N_T)
        sign_u = np.where(u_min_qk > 0, 1.0, -1.0)
        v = gi * (-0.5 * sign_u / (1 + np.abs(u_min_qk))**2)
        r_min_qk = np.where(q_mask, rQ, rK)            # (P, N_T)

        trees_b = np.broadcast_to(trees, (P, N_T))
        ED = x_grad.shape[1]

        # Q gradient scatter -> all go to x_grad[pos]
        if q_mask.any():
            q_t = trees_b[q_mask]
            q_r = r_min_qk[q_mask]
            q_v = v[q_mask]
            x_grad[pos] += np.bincount(lut.a_arr[q_t, q_r], weights=q_v, minlength=ED)
            x_grad[pos] -= np.bincount(lut.b_arr[q_t, q_r], weights=q_v, minlength=ED)

        # K gradient scatter -> x_grad[pos1] for each pair (flat bincount)
        k_mask = ~q_mask
        if k_mask.any():
            k_t = trees_b[k_mask]
            k_r = r_min_qk[k_mask]
            k_v = v[k_mask]
            k_pos1 = np.broadcast_to(np.arange(P)[:, np.newaxis], (P, N_T))[k_mask]
            flat_a = k_pos1 * ED + lut.a_arr[k_t, k_r]
            flat_b = k_pos1 * ED + lut.b_arr[k_t, k_r]
            flat_size = x_grad.size
            x_grad.ravel()[:] += np.bincount(flat_a, weights=k_v, minlength=flat_size)
            x_grad.ravel()[:] -= np.bincount(flat_b, weights=k_v, minlength=flat_size)

        # PE gradient
        abs_uPE = np.abs(uPE)
        pe_mask = (abs_uPE < abs_uQ) & (abs_uPE < abs_uK)
        if pe_mask.any():
            jbarPE_bins = CONCATENATE_vec(jQ, jK, jPE ^ (1 << rPE), args)
            S_jbarPE = _f32(lut.S[trees, jbarPE_bins])  # (P, N_T, y_dim)
            giPE = ((S_jbarPE - S_j) * y_g).sum(axis=2)  # (P, N_T)
            u_pe = uPE[pe_mask]
            sign_pe = np.where(u_pe > 0, 1.0, -1.0)
            deltaPE = giPE[pe_mask] * (-0.5 * sign_pe / (1 + np.abs(u_pe))**2)
            pe_t = trees_b[pe_mask]
            pe_r = rPE[pe_mask]
            pe_pair = np.broadcast_to(np.arange(P)[:, np.newaxis], (P, N_T))[pe_mask]
            np.add.at(pos_grad, (pe_idx[pe_pair], pe_t, pe_r), deltaPE)

        # S update: accumulate for duplicate (tree, bin) across pos1 values
        trees_flat = trees_b.ravel()
        j_flat = j_bins.ravel()
        neg_lr_y_g = np.empty((P * N_T, y_dim), dtype=lut.S.dtype)
        neg_lr_y_g[:] = -(learning_rate * y_g)
        np.add.at(lut.S, (trees_flat, j_flat), neg_lr_y_g)

    head.Positional_encoding -= learning_rate * pos_grad


def model_forward(m, args):
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    for l in range(NUM_LAYERS):
        np.copyto(m._x_buf, m.z)
        for h in range(NUM_HEADS):
            attention_forward(m.head[l][h], m._x_buf, m.z, args)

        for pos in range(args.context_size):
            cache_index(m.FFN[l], m.FFN_cache[l][pos], m.z[pos], args)
            LUT_forward(m.FFN[l], m.FFN_cache[l][pos], m.z[pos], args)

    m.output_head.forward(m.z, args)


def model_backward(m, args):
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

    x_grad = np.zeros((args.context_size, EMBEDDING_DIM), dtype=np.float32)

    m.output_head.backward(x_grad, args)

    for l in range(NUM_LAYERS - 1, -1, -1):
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        for pos in range(args.context_size):
            LUT_backward(m.FFN[l], m.FFN_cache[l][pos], x_grad[pos], y_grad[pos], args)

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
