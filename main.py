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
from word_tokenizer import WordTokenizer
from multiprocessing.pool import ThreadPool
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
    __slots__ = ['V', 'Score', 'Value', 'Positional_encoding',
                 '_v_j', '_v_r_min', '_v_u_min',
                 '_val_j', '_val_r_min', '_val_u_min',
                 '_pe_j', '_pe_r_min', '_pe_u_min',
                 '_attn_w', '_v_vecs']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        POSITIONAL_DIM = args.positional_dim
        ED = args.embedding_dim
        soft = getattr(args, 'soft_attention', False)
        self.Positional_encoding = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)
        self._v_j = np.zeros((CS, N_T), dtype=np.int32)
        self._v_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._v_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._pe_j = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_u_min = np.zeros((CS, N_T), dtype=np.float32)
        if soft:
            self.V = None
            self.Score = LUT(N_T)
            self.Value = LUT(N_T)
            self._val_j     = np.zeros((CS, N_T), dtype=np.int32)
            self._val_r_min = np.zeros((CS, N_T), dtype=np.int32)
            self._val_u_min = np.zeros((CS, N_T), dtype=np.float32)
            self._attn_w    = np.zeros((CS, CS),  dtype=np.float32)
            self._v_vecs    = np.zeros((CS, ED),  dtype=np.float32)
        else:
            self.V = LUT(N_T)
            self.Score = None
            self.Value = None
            self._val_j     = None
            self._val_r_min = None
            self._val_u_min = None
            self._attn_w    = None
            self._v_vecs    = None


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
        lut = self.unembedder
        cache_index_batch(lut, z, self._j, self._r_min, self._u_min)
        if HAS_NUMBA and lut.S.dtype == np.float32:
            _lut_forward_kernel(lut.S, lut.trees, self._j, self.output,
                                args.context_size, args.n_t, lut.y_dim)
        else:
            self.output[:] = 0.0
            self.output[:, :lut.y_dim] += _f32(lut.S[lut.trees, self._j]).sum(axis=1)

    def backward(self, x_grad, args, skip_s_update=False):
        global learning_rate
        lut = self.unembedder
        if HAS_NUMBA and lut.S.dtype == np.float32:
            if skip_s_update:
                _lut_backward_kernel_noupdate(lut.S, lut.trees, self._j, self._r_min, self._u_min,
                                              self.output, x_grad, lut.a_arr, lut.b_arr,
                                              lut.y_dim, args.context_size, args.n_t)
            else:
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
                if not skip_s_update:
                    lut.S[trees, j_bins] -= learning_rate * self.output[pos, :lut.y_dim]

    def compute_gradients(self, args):
        if HAS_NUMBA:
            _softmax_cross_entropy_kernel(self.output, self.tokens[1:args.context_size + 1],
                                          args.context_size, args.vocab_size)
        else:
            x = self.output
            x -= x.max(axis=1, keepdims=True)
            np.exp(x, out=x)
            x /= x.sum(axis=1, keepdims=True)
            x[np.arange(args.context_size), self.tokens[1:args.context_size + 1]] -= 1.0

    def sample_token(self, args):
        VOCAB_SIZE = args.vocab_size
        softmax(self.output[args.context_size - 1], VOCAB_SIZE, args.temperature)
        return sample(self.output[args.context_size - 1], VOCAB_SIZE)

    def training_loss(self, args):
        """Mean cross-entropy over all context positions from raw logits."""
        CS = args.context_size
        x = self.output[:CS]
        targets = self.tokens[1:CS + 1]
        x_max = x.max(axis=1)
        log_sum_exp = x_max + np.log(np.exp(x - x_max[:, None]).sum(axis=1))
        return float((log_sum_exp - x[np.arange(CS), targets]).mean())

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
        CS = args.context_size
        NT = args.n_t
        cache_index_batch(self.unembedder_hi, z, self._hi_j, self._hi_r_min, self._hi_u_min)
        cache_index_batch(self.unembedder_lo, z, self._lo_j, self._lo_r_min, self._lo_u_min)
        if HAS_NUMBA and self.unembedder_hi.S.dtype == np.float32:
            _lut_forward_kernel(self.unembedder_hi.S, self.unembedder_hi.trees, self._hi_j,
                                self.output_hi, CS, NT, self.unembedder_hi.y_dim)
            _lut_forward_kernel(self.unembedder_lo.S, self.unembedder_lo.trees, self._lo_j,
                                self.output_lo, CS, NT, self.unembedder_lo.y_dim)
        else:
            self.output_hi[:] = 0.0
            self.output_lo[:] = 0.0
            self.output_hi[:, :self.unembedder_hi.y_dim] += _f32(self.unembedder_hi.S[self.unembedder_hi.trees, self._hi_j]).sum(axis=1)
            self.output_lo[:, :self.unembedder_lo.y_dim] += _f32(self.unembedder_lo.S[self.unembedder_lo.trees, self._lo_j]).sum(axis=1)

    def backward(self, x_grad, args, skip_s_update=False):
        global learning_rate
        lut_list = [
            (self.unembedder_hi, self._hi_j, self._hi_r_min, self._hi_u_min, self.output_hi),
            (self.unembedder_lo, self._lo_j, self._lo_r_min, self._lo_u_min, self.output_lo),
        ]
        for lut, j, r_min, u_min, output in lut_list:
            if HAS_NUMBA and lut.S.dtype == np.float32:
                if skip_s_update:
                    _lut_backward_kernel_noupdate(lut.S, lut.trees, j, r_min, u_min,
                                                  output, x_grad, lut.a_arr, lut.b_arr,
                                                  lut.y_dim, args.context_size, args.n_t)
                else:
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
                    if not skip_s_update:
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

    def training_loss(self, args):
        """Mean cross-entropy over all context positions (hi + lo components)."""
        CS = args.context_size
        loss = 0.0
        for x, targets in [(self.output_hi[:CS], self.tokens_hi[1:CS + 1]),
                            (self.output_lo[:CS], self.tokens_lo[1:CS + 1])]:
            x_max = x.max(axis=1)
            log_sum_exp = x_max + np.log(np.exp(x - x_max[:, None]).sum(axis=1))
            loss += float((log_sum_exp - x[np.arange(CS), targets]).mean())
        return loss

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
                 '_ffn_j', '_ffn_r_min', '_ffn_u_min', 'head', 'output_head',
                 'ln_attn_gamma', 'ln_attn_beta', 'ln_ffn_gamma', 'ln_ffn_beta',
                 '_ln_attn_xhat', '_ln_attn_rstd', '_ln_ffn_xhat', '_ln_ffn_rstd',
                 '_ln_ffn_buf', '_drop_attn_mask', '_drop_ffn_mask']
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

        # LayerNorm parameters (NL x ED)
        self.ln_attn_gamma = np.ones((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        self.ln_attn_beta  = np.zeros((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        self.ln_ffn_gamma  = np.ones((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        self.ln_ffn_beta   = np.zeros((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        # LN caches
        self._ln_attn_xhat = [np.zeros((CS, EMBEDDING_DIM), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_attn_rstd = [np.zeros((CS, 1), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_ffn_xhat  = [np.zeros((CS, EMBEDDING_DIM), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_ffn_rstd  = [np.zeros((CS, 1), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_ffn_buf   = np.zeros((CS, EMBEDDING_DIM), dtype=np.float32)
        # Dropout masks
        self._drop_attn_mask = [None] * NUM_LAYERS
        self._drop_ffn_mask  = [None] * NUM_LAYERS


class TrainingData:
    __slots__ = ['data', 'length', 'val_data', 'val_length', 'testing_input_data']
    def __init__(self):
        self.data = None
        self.length = 0
        self.val_data = None
        self.val_length = 0
        self.testing_input_data = None


class SparseGradAccumulator:
    """Lightweight per-batch-element gradient storage (~1 MB vs ~3 GB for dense).

    Instead of allocating full-size S gradient buffers, we save only the small
    y_grad arrays during backward.  The actual S updates are replayed at apply
    time using the saved y_grads together with the cached index arrays that
    already live on each BatchElement.
    """
    __slots__ = ['ffn_y_grads', 'attn_y_grads', 'pe_grads',
                 'attn_score_grads', 'attn_val_grads',
                 'ln_attn_dgamma', 'ln_attn_dbeta', 'ln_ffn_dgamma', 'ln_ffn_dbeta']

    def __init__(self, args):
        NL = args.num_layers
        NH = args.num_heads
        CS = args.context_size
        NT = args.n_t
        PD = args.positional_dim
        ED = args.embedding_dim
        self.ffn_y_grads = [None] * NL      # filled during backward
        self.attn_y_grads = [None] * NL     # filled during backward
        self.pe_grads = [[np.zeros((CS, NT, PD), dtype=np.float32)
                          for _ in range(NH)] for _ in range(NL)]
        if getattr(args, 'soft_attention', False):
            self.attn_score_grads = [[np.zeros((CS, CS), dtype=np.float32)
                                      for _ in range(NH)] for _ in range(NL)]
            self.attn_val_grads   = [[np.zeros((CS, ED), dtype=np.float32)
                                      for _ in range(NH)] for _ in range(NL)]
        else:
            self.attn_score_grads = None
            self.attn_val_grads   = None
        self.ln_attn_dgamma = np.zeros((NL, ED), dtype=np.float32)
        self.ln_attn_dbeta  = np.zeros((NL, ED), dtype=np.float32)
        self.ln_ffn_dgamma  = np.zeros((NL, ED), dtype=np.float32)
        self.ln_ffn_dbeta   = np.zeros((NL, ED), dtype=np.float32)

    def zero(self):
        for i in range(len(self.ffn_y_grads)):
            self.ffn_y_grads[i] = None
        for i in range(len(self.attn_y_grads)):
            self.attn_y_grads[i] = None
        if self.attn_score_grads is not None:
            for layer in self.attn_score_grads:
                for g in layer:
                    g[:] = 0
            for layer in self.attn_val_grads:
                for g in layer:
                    g[:] = 0
        for layer in self.pe_grads:
            for g in layer:
                g[:] = 0
        self.ln_attn_dgamma[:] = 0
        self.ln_attn_dbeta[:] = 0
        self.ln_ffn_dgamma[:] = 0
        self.ln_ffn_dbeta[:] = 0


def _share_lut(dst, src):
    """Make dst LUT share src's S table and parameters."""
    dst.S = src.S
    dst.y_dim = src.y_dim
    dst.a_arr = src.a_arr
    dst.b_arr = src.b_arr
    dst.bitmasks = src.bitmasks
    dst.trees = src.trees


class BatchElement:
    """Per-sequence scratch space that duck-types as Model, sharing S tables."""
    __slots__ = ['Token_embedder', 'tokens', 'z', '_x_buf', 'FFN',
                 '_ffn_j', '_ffn_r_min', '_ffn_u_min', 'head', 'output_head',
                 'grad_accum',
                 'ln_attn_gamma', 'ln_attn_beta', 'ln_ffn_gamma', 'ln_ffn_beta',
                 '_ln_attn_xhat', '_ln_attn_rstd', '_ln_ffn_xhat', '_ln_ffn_rstd',
                 '_ln_ffn_buf', '_drop_attn_mask', '_drop_ffn_mask']

    def __init__(self, m, args):
        CS = args.context_size
        ED = args.embedding_dim
        NT = args.n_t
        NL = args.num_layers
        NH = args.num_heads

        # Shared (read-only during batch processing)
        self.Token_embedder = m.Token_embedder
        self.FFN = m.FFN  # shared LUT objects (S tables, a/b arrays)
        # Shared LN parameters
        self.ln_attn_gamma = m.ln_attn_gamma
        self.ln_attn_beta  = m.ln_attn_beta
        self.ln_ffn_gamma  = m.ln_ffn_gamma
        self.ln_ffn_beta   = m.ln_ffn_beta

        # Per-element scratch
        self.tokens = np.zeros(CS + 1, dtype=np.int32)
        self.z = np.zeros((CS, ED), dtype=np.float32)
        self._x_buf = np.zeros_like(self.z)
        self._ffn_j = [np.zeros((CS, NT), dtype=np.int32) for _ in range(NL)]
        self._ffn_r_min = [np.zeros((CS, NT), dtype=np.int32) for _ in range(NL)]
        self._ffn_u_min = [np.zeros((CS, NT), dtype=np.float32) for _ in range(NL)]

        # Per-element attention heads (own scratch, shared LUTs and PE)
        self.head = [[AttentionHead(args) for _ in range(NH)] for _ in range(NL)]
        for l in range(NL):
            for h in range(NH):
                if getattr(args, 'soft_attention', False):
                    _share_lut(self.head[l][h].Score, m.head[l][h].Score)
                    _share_lut(self.head[l][h].Value, m.head[l][h].Value)
                else:
                    _share_lut(self.head[l][h].V, m.head[l][h].V)
                self.head[l][h].Positional_encoding = m.head[l][h].Positional_encoding

        # Per-element output head (own output buffers, shared LUT)
        if args.factored_output:
            self.output_head = FactoredOutputHead(args)
            _share_lut(self.output_head.unembedder_hi, m.output_head.unembedder_hi)
            _share_lut(self.output_head.unembedder_lo, m.output_head.unembedder_lo)
        else:
            self.output_head = StandardOutputHead(args)
            _share_lut(self.output_head.unembedder, m.output_head.unembedder)

        # Per-element LN caches and dropout masks
        self._ln_attn_xhat = [np.zeros((CS, ED), np.float32) for _ in range(NL)]
        self._ln_attn_rstd = [np.zeros((CS, 1), np.float32) for _ in range(NL)]
        self._ln_ffn_xhat  = [np.zeros((CS, ED), np.float32) for _ in range(NL)]
        self._ln_ffn_rstd  = [np.zeros((CS, 1), np.float32) for _ in range(NL)]
        self._ln_ffn_buf   = np.zeros((CS, ED), dtype=np.float32)
        self._drop_attn_mask = [None] * NL
        self._drop_ffn_mask  = [None] * NL

        # Sparse gradient accumulators (saves y_grads, not full S buffers)
        self.grad_accum = SparseGradAccumulator(args)


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
    if HAS_NUMBA and x_batch.dtype == np.float32:
        _cache_index_kernel(x_batch, lut.a_arr, lut.b_arr, lut.bitmasks,
                            j_out, r_min_out, u_min_out,
                            x_batch.shape[0], lut.a_arr.shape[0], lut.a_arr.shape[1])
    else:
        u = x_batch[:, lut.a_arr] - x_batch[:, lut.b_arr]      # (CS, N_T, N_C)
        j_out[:] = ((u > 0).astype(np.int32) * lut.bitmasks).sum(axis=2)
        abs_u = np.abs(u)
        r_min_out[:] = abs_u.argmin(axis=2)
        cs_idx = np.arange(x_batch.shape[0])[:, np.newaxis]
        u_min_out[:] = u[cs_idx, lut.trees, r_min_out]


def cache_PE_index_batch(u_batch, j_out, r_min_out, u_min_out, args):
    """Batch cache_PE_index across all positions. u_batch: (CS, N_T, PD), outputs: (CS, N_T)."""
    if HAS_NUMBA and u_batch.dtype == np.float32:
        _cache_pe_index_kernel(u_batch, args.pe_bitmasks,
                               j_out, r_min_out, u_min_out,
                               u_batch.shape[0], u_batch.shape[1], u_batch.shape[2])
    else:
        j_out[:] = ((u_batch > 0).astype(np.int32) * args.pe_bitmasks).sum(axis=2)
        abs_u = np.abs(u_batch)
        r_min_out[:] = abs_u.argmin(axis=2)
        cs_idx = np.arange(u_batch.shape[0])[:, np.newaxis]
        tree_idx = np.arange(u_batch.shape[1])
        u_min_out[:] = u_batch[cs_idx, tree_idx, r_min_out]


def _f32(arr):
    """Cast to float32 if needed (no-op if already float32)."""
    return arr if arr.dtype == np.float32 else arr.astype(np.float32)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x: (CS, ED). Returns (y, xhat, rstd)."""
    mean = x.mean(axis=1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(axis=1, keepdims=True) + eps)
    xhat = (x - mean) * rstd
    return xhat * gamma + beta, xhat, rstd


def layernorm_backward(dy, xhat, rstd, gamma):
    """Returns (dx, dgamma, dbeta)."""
    N = dy.shape[1]
    dgamma = (dy * xhat).sum(axis=0)
    dbeta  = dy.sum(axis=0)
    dx = (gamma * rstd / N) * (
        N * dy
        - dy.sum(axis=1, keepdims=True)
        - xhat * (dy * xhat).sum(axis=1, keepdims=True)
    )
    return dx, dgamma, dbeta


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


@njit(parallel=True, cache=True)
def _attention_backward_kernel_noupdate(S, all_jV, all_rV, all_uV, all_jPE, all_rPE, all_uPE,
                                         y_grad, x_grad, pos_grad, a_arr, b_arr,
                                         shift_qk, pd, CS, N_T, y_dim):
    """Like _attention_backward_kernel but without Phase 3 (no S update)."""
    ED = x_grad.shape[1]
    n_threads = get_num_threads()
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
                    x_grad[pos, idx_a] += v
                    x_grad[pos, idx_b] -= v
                else:
                    x_grad_thr[tid, pos1, idx_a] += v
                    x_grad_thr[tid, pos1, idx_b] -= v

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


@njit(parallel=True, cache=True)
def _cache_index_kernel(x, a_arr, b_arr, bitmasks, j_out, r_min_out, u_min_out, CS, N_T, N_C):
    """Parallel cache_index: computes j, r_min, u_min for all (pos, tree) pairs."""
    for pos in prange(CS):
        for t in range(N_T):
            j = np.int32(0)
            min_abs_u = np.float32(1e38)
            r_min = np.int32(0)
            u_min_val = np.float32(0.0)
            for c in range(N_C):
                u_val = x[pos, a_arr[t, c]] - x[pos, b_arr[t, c]]
                if u_val > np.float32(0.0):
                    j |= bitmasks[c]
                au = abs(u_val)
                if au < min_abs_u:
                    min_abs_u = au
                    r_min = np.int32(c)
                    u_min_val = u_val
            j_out[pos, t] = j
            r_min_out[pos, t] = r_min
            u_min_out[pos, t] = u_min_val


@njit(parallel=True, cache=True)
def _cache_pe_index_kernel(u_batch, pe_bitmasks, j_out, r_min_out, u_min_out, CS, N_T, PD):
    """Parallel cache_PE_index: computes j, r_min, u_min from PE tensor."""
    for pos in prange(CS):
        for t in range(N_T):
            j = np.int32(0)
            min_abs_u = np.float32(1e38)
            r_min = np.int32(0)
            u_min_val = np.float32(0.0)
            for p in range(PD):
                u_val = u_batch[pos, t, p]
                if u_val > np.float32(0.0):
                    j |= pe_bitmasks[p]
                au = abs(u_val)
                if au < min_abs_u:
                    min_abs_u = au
                    r_min = np.int32(p)
                    u_min_val = u_val
            j_out[pos, t] = j
            r_min_out[pos, t] = r_min
            u_min_out[pos, t] = u_min_val


@njit(parallel=True, cache=True)
def _lut_forward_add_kernel(S, trees, j, output, CS, N_T, y_dim):
    """Fused LUT forward that adds to output (no zeroing). Used for FFN residual."""
    for pos in prange(CS):
        for t in range(N_T):
            j_bin = j[pos, t]
            for d in range(y_dim):
                output[pos, d] += S[trees[t], j_bin, d]


@njit(parallel=True, cache=True)
def _lut_forward_kernel(S, trees, j, output, CS, N_T, y_dim):
    """Fused LUT forward: parallel over CS, eliminates (CS, N_T, y_dim) temporary."""
    for pos in prange(CS):
        for d in range(y_dim):
            output[pos, d] = np.float32(0.0)
        for t in range(N_T):
            j_bin = j[pos, t]
            for d in range(y_dim):
                output[pos, d] += S[trees[t], j_bin, d]


@njit(parallel=True, cache=True)
def _lut_backward_kernel(S, trees, j, r_min, u_min, grad, x_grad, a_arr, b_arr,
                          y_dim, CS, N_T, lr):
    """Fused backward for simple LUT (FFN, output head). grad: (CS, >=y_dim).

    Phase 1 (parallel): compute x_grad using original S — each pos writes unique row.
    Phase 2 (sequential): update S — bin collisions possible across positions.
    """
    # Phase 1: parallel x_grad computation
    for pos in prange(CS):
        for t in range(N_T):
            tree = trees[t]
            j_bin = j[pos, t]
            r = r_min[pos, t]
            jbar_bin = j_bin ^ (1 << r)
            gi = np.float32(0.0)
            for d in range(y_dim):
                gi += (S[tree, jbar_bin, d] - S[tree, j_bin, d]) * grad[pos, d]
            u = u_min[pos, t]
            sign_u = np.float32(1.0) if u > np.float32(0.0) else np.float32(-1.0)
            v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u)) ** 2)
            x_grad[pos, a_arr[t, r]] += v
            x_grad[pos, b_arr[t, r]] -= v

    # Phase 2: sequential S update
    for pos in range(CS):
        for t in range(N_T):
            tree = trees[t]
            j_bin = j[pos, t]
            for d in range(y_dim):
                S[tree, j_bin, d] -= lr * grad[pos, d]


@njit(parallel=True, cache=True)
def _lut_backward_kernel_noupdate(S, trees, j, r_min, u_min, grad, x_grad, a_arr, b_arr,
                                   y_dim, CS, N_T):
    """Like _lut_backward_kernel but without the S update: fully parallel over CS."""
    for pos in prange(CS):
        for t in range(N_T):
            tree = trees[t]
            j_bin = j[pos, t]
            r = r_min[pos, t]
            jbar_bin = j_bin ^ (1 << r)
            gi = np.float32(0.0)
            for d in range(y_dim):
                gi += (S[tree, jbar_bin, d] - S[tree, j_bin, d]) * grad[pos, d]
            u = u_min[pos, t]
            sign_u = np.float32(1.0) if u > np.float32(0.0) else np.float32(-1.0)
            v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u)) ** 2)
            x_grad[pos, a_arr[t, r]] += v
            x_grad[pos, b_arr[t, r]] -= v


@njit(cache=True)
def _lut_s_replay_kernel(S, trees, j, y_grad, y_dim, CS, N_T, scale):
    """Replay LUT S updates from saved y_grad and cached j indices."""
    for pos in range(CS):
        for t in range(N_T):
            tree = trees[t]
            j_bin = j[pos, t]
            for d in range(y_dim):
                S[tree, j_bin, d] -= scale * y_grad[pos, d]


@njit(cache=True)
def _attention_s_replay_kernel(S, all_jV, all_jPE, y_grad, shift_qk, pd, CS, N_T, y_dim, scale):
    """Replay attention S updates from saved y_grad and cached j indices."""
    for pos in range(1, CS):
        for pos1 in range(pos):
            for t in range(N_T):
                j_bin = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pos - pos1, t]
                for d in range(y_dim):
                    S[t, j_bin, d] -= scale * y_grad[pos, d]


@njit(parallel=True, cache=True)
def _softmax_cross_entropy_kernel(output, target_indices, CS, VOCAB):
    """Parallel softmax + subtract one-hot over all context positions."""
    for pos in prange(CS):
        max_val = output[pos, 0]
        for v in range(1, VOCAB):
            if output[pos, v] > max_val:
                max_val = output[pos, v]
        s = np.float32(0.0)
        for v in range(VOCAB):
            output[pos, v] = np.exp(output[pos, v] - max_val)
            s += output[pos, v]
        for v in range(VOCAB):
            output[pos, v] /= s
        output[pos, target_indices[pos]] -= np.float32(1.0)


# ---------------------------------------------------------------------------
# Soft attention Numba kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _soft_attn_forward_kernel(ScoreS, ValueS, jV, val_j, jPE, y, attn_w, v_vecs,
                               shift_qk, pd, CS, N_T, ED):
    """Fused soft attention forward: raw scores, softmax, value vectors, weighted sum."""
    # Phase 1: raw scores + softmax (parallel over pos >= 1)
    for pos in prange(1, CS):
        for pos1 in range(pos):
            pe_pos = pos - pos1
            s = np.float32(0.0)
            for t in range(N_T):
                j_bin = (jV[pos, t] << shift_qk) | (jV[pos1, t] << pd) | jPE[pe_pos, t]
                s += ScoreS[t, j_bin, 0]
            attn_w[pos, pos1] = s
        max_s = attn_w[pos, 0]
        for pos1 in range(1, pos):
            if attn_w[pos, pos1] > max_s:
                max_s = attn_w[pos, pos1]
        s_sum = np.float32(0.0)
        for pos1 in range(pos):
            val = np.exp(attn_w[pos, pos1] - max_s)
            attn_w[pos, pos1] = val
            s_sum += val
        for pos1 in range(pos):
            attn_w[pos, pos1] /= s_sum

    # Phase 2: value vectors v_vecs[pos] = sum_t ValueS[t, val_j[pos,t], :] (parallel over pos)
    for pos in prange(CS):
        for d in range(ED):
            v_vecs[pos, d] = np.float32(0.0)
        for t in range(N_T):
            j_val = val_j[pos, t]
            for d in range(ED):
                v_vecs[pos, d] += ValueS[t, j_val, d]

    # Phase 3: weighted sum into y (parallel over pos >= 1)
    for pos in prange(1, CS):
        for pos1 in range(pos):
            w = attn_w[pos, pos1]
            for d in range(ED):
                y[pos, d] += w * v_vecs[pos1, d]


@njit(parallel=True, cache=True)
def _soft_attn_backward_value_kernel(ValueS, val_j, val_r_min, val_u_min,
                                      attn_w, y_grad, val_grad, x_grad,
                                      val_a_arr, val_b_arr, CS, N_T, ED, lr, skip_s_update):
    """Soft attention value backward: val_grad, STE x_grad update, optional Value.S update."""
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)

    # Phase 1: val_grad[pos1] = sum_{pos>pos1} attn_w[pos,pos1] * y_grad[pos] (parallel over pos1)
    for pos1 in prange(CS):
        for d in range(ED):
            val_grad[pos1, d] = np.float32(0.0)
        for pos in range(pos1 + 1, CS):
            w = attn_w[pos, pos1]
            for d in range(ED):
                val_grad[pos1, d] += w * y_grad[pos, d]

    # Phase 2: Value STE — compute x_grad contribution (parallel over pos1)
    for pos1 in prange(CS):
        tid = get_thread_id()
        for t in range(N_T):
            j_val = val_j[pos1, t]
            r = val_r_min[pos1, t]
            jbar = j_val ^ (1 << r)
            gi = np.float32(0.0)
            for d in range(ED):
                gi += (ValueS[t, jbar, d] - ValueS[t, j_val, d]) * val_grad[pos1, d]
            u = val_u_min[pos1, t]
            sign_u = np.float32(1.0) if u > np.float32(0.0) else np.float32(-1.0)
            v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u)) ** 2)
            x_grad_thr[tid, pos1, val_a_arr[t, r]] += v
            x_grad_thr[tid, pos1, val_b_arr[t, r]] -= v

    # Phase 3: reduce x_grad_thr into x_grad (parallel over pos)
    for pos in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, pos, d]
            x_grad[pos, d] += total

    # Phase 4: Value.S update (sequential — collisions possible across pos1)
    if not skip_s_update:
        for pos1 in range(CS):
            for t in range(N_T):
                j_val = val_j[pos1, t]
                for d in range(ED):
                    ValueS[t, j_val, d] -= lr * val_grad[pos1, d]


@njit(parallel=True, cache=True)
def _soft_attn_backward_score_kernel(ScoreS, jV, rV, uV, jPE, rPE, uPE,
                                      attn_w, v_vecs, y_grad, score_grad, x_grad, pos_grad,
                                      score_a_arr, score_b_arr,
                                      shift_qk, pd, CS, N_T, ED, lr, skip_s_update):
    """Soft attention score backward: score_grad, STE x_grad + pe_grad, optional Score.S update."""
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)
    pe_grad_thr = np.zeros((n_threads, CS, N_T, pd), dtype=np.float32)

    # Phase 1: compute score_grad (parallel over pos >= 1)
    # score_grad[pos, pos1] = attn_w[pos,pos1] * (a[pos,pos1] - baseline[pos])
    # Uses score_grad as temp for affinities, then overwrites with score_grad.
    for pos in prange(1, CS):
        baseline = np.float32(0.0)
        for pos1 in range(pos):
            a = np.float32(0.0)
            for d in range(ED):
                a += y_grad[pos, d] * v_vecs[pos1, d]
            score_grad[pos, pos1] = a
            baseline += attn_w[pos, pos1] * a
        for pos1 in range(pos):
            score_grad[pos, pos1] = attn_w[pos, pos1] * (score_grad[pos, pos1] - baseline)

    # Phase 2: Score STE — x_grad and pe_grad contributions (parallel over pos >= 1)
    for pos in prange(1, CS):
        tid = get_thread_id()
        for pos1 in range(pos):
            sg = score_grad[pos, pos1]
            pe_pos = pos - pos1
            for t in range(N_T):
                jQ_t  = jV[pos, t];  jK_t = jV[pos1, t];  jPE_t = jPE[pe_pos, t]
                j_bin = (jQ_t << shift_qk) | (jK_t << pd) | jPE_t
                abs_uQ  = abs(uV[pos, t])
                abs_uK  = abs(uV[pos1, t])
                abs_uPE = abs(uPE[pe_pos, t])
                q_wins = abs_uQ < abs_uK
                if q_wins:
                    flip_bit = rV[pos, t] + shift_qk
                    u_min = uV[pos, t];  r = rV[pos, t]
                else:
                    flip_bit = rV[pos1, t] + pd
                    u_min = uV[pos1, t];  r = rV[pos1, t]
                jbar = j_bin ^ (1 << flip_bit)
                gi = (ScoreS[t, jbar, 0] - ScoreS[t, j_bin, 0]) * sg
                sign_u = np.float32(1.0) if u_min > np.float32(0.0) else np.float32(-1.0)
                v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u_min)) ** 2)
                idx_a = score_a_arr[t, r];  idx_b = score_b_arr[t, r]
                if q_wins:
                    x_grad[pos, idx_a] += v   # unique per prange pos — no race
                    x_grad[pos, idx_b] -= v
                else:
                    x_grad_thr[tid, pos1, idx_a] += v
                    x_grad_thr[tid, pos1, idx_b] -= v
                # PE branch
                if abs_uPE < abs_uQ and abs_uPE < abs_uK:
                    rPE_t = rPE[pe_pos, t]
                    jbarPE = j_bin ^ (1 << rPE_t)
                    giPE = (ScoreS[t, jbarPE, 0] - ScoreS[t, j_bin, 0]) * sg
                    u_pe = uPE[pe_pos, t]
                    sign_pe = np.float32(1.0) if u_pe > np.float32(0.0) else np.float32(-1.0)
                    deltaPE = giPE * (np.float32(-0.5) * sign_pe / (np.float32(1.0) + abs(u_pe)) ** 2)
                    pe_grad_thr[tid, pe_pos, t, rPE_t] += deltaPE

    # Phase 3: reduce thread-local buffers (parallel)
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

    # Phase 4: Score.S update (sequential — collisions possible across (pos,pos1) pairs)
    if not skip_s_update:
        for pos in range(1, CS):
            for pos1 in range(pos):
                sg = score_grad[pos, pos1]
                pe_pos = pos - pos1
                for t in range(N_T):
                    j_bin = (jV[pos, t] << shift_qk) | (jV[pos1, t] << pd) | jPE[pe_pos, t]
                    ScoreS[t, j_bin, 0] -= lr * sg


@njit(cache=True)
def _soft_attn_score_s_replay_kernel(ScoreS, jV, jPE, score_grad, shift_qk, pd, CS, N_T, scale):
    """Replay Score.S updates from saved score_grad and cached j indices."""
    for pos in range(1, CS):
        for pos1 in range(pos):
            sg = score_grad[pos, pos1]
            pe_pos = pos - pos1
            for t in range(N_T):
                j_bin = (jV[pos, t] << shift_qk) | (jV[pos1, t] << pd) | jPE[pe_pos, t]
                ScoreS[t, j_bin, 0] -= scale * sg


@njit(cache=True)
def _soft_attn_value_s_replay_kernel(ValueS, val_j, val_grad, CS, N_T, ED, scale):
    """Replay Value.S updates from saved val_grad and cached j indices."""
    for pos1 in range(CS):
        for t in range(N_T):
            j_val = val_j[pos1, t]
            for d in range(ED):
                ValueS[t, j_val, d] -= scale * val_grad[pos1, d]


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
            if getattr(args, 'soft_attention', False):
                build_LUT(m.head[l][h].Score, N_C + N_C + POSITIONAL_DIM, 1, args)
                build_LUT(m.head[l][h].Value, N_C, EMBEDDING_DIM, args)
            else:
                build_LUT(m.head[l][h].V, N_C + N_C + POSITIONAL_DIM, EMBEDDING_DIM, args)

    m.output_head.build(N_C, args)


def attention_forward(head, x, y, args):
    CS = args.context_size

    if getattr(args, 'soft_attention', False):
        shift_qk = args.shift_qk
        pd = args.positional_dim
        ED = args.embedding_dim

        # Two independent cache calls (Score: Q+K indexing; Value: K-only indexing)
        cache_index_batch(head.Score, x, head._v_j, head._v_r_min, head._v_u_min)
        cache_index_batch(head.Value, x, head._val_j, head._val_r_min, head._val_u_min)
        cache_PE_index_batch(head.Positional_encoding, head._pe_j, head._pe_r_min, head._pe_u_min, args)

        if HAS_NUMBA and head.Score.S.dtype == np.float32:
            _soft_attn_forward_kernel(head.Score.S, head.Value.S,
                                       head._v_j, head._val_j, head._pe_j,
                                       y, head._attn_w, head._v_vecs,
                                       shift_qk, pd, CS, args.n_t, ED)
        else:
            # Value vectors: v_vecs[pos] = sum_t Value.S[t, val_j[pos, t], :]
            val_trees = head.Value.trees
            v_vecs = _f32(head.Value.S[val_trees, head._val_j]).sum(axis=1)  # (CS, ED)
            head._v_vecs[:] = v_vecs

            # Raw scores for all causal (pos, pos1) pairs
            score_trees = head.Score.trees
            raw_scores = np.full((CS, CS), -1e9, dtype=np.float32)
            for pos in range(1, CS):
                jQ  = head._v_j[pos]                              # (N_T,)
                jK  = head._v_j[:pos]                             # (pos, N_T)
                jPE = head._pe_j[pos - np.arange(pos)]            # (pos, N_T)
                j_bins = (jQ << shift_qk) | (jK << pd) | jPE     # (pos, N_T)
                raw_scores[pos, :pos] = _f32(head.Score.S[score_trees, j_bins, 0]).sum(axis=1)

            # Softmax + weighted sum
            for pos in range(1, CS):
                s = raw_scores[pos, :pos].copy()
                s -= s.max()
                np.exp(s, out=s)
                s /= s.sum()
                head._attn_w[pos, :pos] = s
                y[pos, :ED] += (s[:, np.newaxis] * v_vecs[:pos]).sum(axis=0)
        return

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


def attention_backward(head, x_grad, y_grad, args, skip_s_update=False, pe_grad_out=None,
                       score_grad_out=None, val_grad_out=None):
    global learning_rate
    CS = args.context_size
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim

    if getattr(args, 'soft_attention', False):
        shift_qk = args.shift_qk
        pd = POSITIONAL_DIM
        ED = x_grad.shape[1]
        val_lut = head.Value
        score_lut = head.Score

        if HAS_NUMBA and score_lut.S.dtype == np.float32:
            val_grad   = val_grad_out   if val_grad_out   is not None else np.zeros((CS, ED),              dtype=np.float32)
            score_grad = score_grad_out if score_grad_out is not None else np.zeros((CS, CS),              dtype=np.float32)
            pos_grad   = pe_grad_out    if pe_grad_out    is not None else np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

            _soft_attn_backward_value_kernel(
                val_lut.S, head._val_j, head._val_r_min, head._val_u_min,
                head._attn_w, y_grad, val_grad, x_grad,
                val_lut.a_arr, val_lut.b_arr,
                CS, N_T, ED, np.float32(learning_rate), skip_s_update)

            _soft_attn_backward_score_kernel(
                score_lut.S, head._v_j, head._v_r_min, head._v_u_min,
                head._pe_j, head._pe_r_min, head._pe_u_min,
                head._attn_w, head._v_vecs, y_grad, score_grad, x_grad, pos_grad,
                score_lut.a_arr, score_lut.b_arr,
                shift_qk, pd, CS, N_T, ED, np.float32(learning_rate), skip_s_update)

            if pe_grad_out is None:
                head.Positional_encoding -= learning_rate * pos_grad
        else:
            val_trees = val_lut.trees
            score_trees = score_lut.trees

            # --- Value path ---
            val_grad = (_f32(head._attn_w)[:, :, np.newaxis] * y_grad[:, np.newaxis, :]).sum(axis=0)

            jbar_val = head._val_j ^ (1 << head._val_r_min)
            S_j_val    = _f32(val_lut.S[val_trees, head._val_j])
            S_jbar_val = _f32(val_lut.S[val_trees, jbar_val])
            gi_val = ((S_jbar_val - S_j_val) * val_grad[:, np.newaxis, :]).sum(axis=2)
            sign_u_val = np.where(head._val_u_min > 0, 1.0, -1.0)
            v_val = gi_val * (-0.5 * sign_u_val / (1 + np.abs(head._val_u_min))**2)

            pos_idx = np.arange(CS)[:, np.newaxis]
            idx_a_val = val_lut.a_arr[val_trees, head._val_r_min]
            idx_b_val = val_lut.b_arr[val_trees, head._val_r_min]
            flat_a_val = (pos_idx * ED + idx_a_val).ravel()
            flat_b_val = (pos_idx * ED + idx_b_val).ravel()
            flat_size = x_grad.size
            x_grad.ravel()[:] += np.bincount(flat_a_val, weights=v_val.ravel(), minlength=flat_size)
            x_grad.ravel()[:] -= np.bincount(flat_b_val, weights=v_val.ravel(), minlength=flat_size)

            if not skip_s_update:
                for pos1 in range(CS):
                    val_lut.S[val_trees, head._val_j[pos1]] -= learning_rate * val_grad[pos1]

            if val_grad_out is not None:
                val_grad_out[:] = val_grad

            # --- Score path ---
            a = y_grad @ head._v_vecs.T
            baseline = (head._attn_w * a).sum(axis=1, keepdims=True)
            score_grad = head._attn_w * (a - baseline)

            pos_grad = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

            for pos in range(1, CS):
                for pos1 in range(pos):
                    sg = score_grad[pos, pos1]
                    pe_pos = pos - pos1
                    for t in range(N_T):
                        jQ_t  = int(head._v_j[pos, t])
                        jK_t  = int(head._v_j[pos1, t])
                        jPE_t = int(head._pe_j[pe_pos, t])
                        j_bin = (jQ_t << shift_qk) | (jK_t << pd) | jPE_t

                        abs_uQ  = abs(float(head._v_u_min[pos, t]))
                        abs_uK  = abs(float(head._v_u_min[pos1, t]))
                        abs_uPE = abs(float(head._pe_u_min[pe_pos, t]))

                        q_wins = abs_uQ < abs_uK
                        if q_wins:
                            flip_bit = int(head._v_r_min[pos, t]) + shift_qk
                            u_min = float(head._v_u_min[pos, t])
                            r = int(head._v_r_min[pos, t])
                        else:
                            flip_bit = int(head._v_r_min[pos1, t]) + pd
                            u_min = float(head._v_u_min[pos1, t])
                            r = int(head._v_r_min[pos1, t])

                        jbar = j_bin ^ (1 << flip_bit)
                        gi = (float(score_lut.S[t, jbar, 0]) - float(score_lut.S[t, j_bin, 0])) * sg
                        sign_u = 1.0 if u_min > 0 else -1.0
                        v = gi * (-0.5 * sign_u / (1.0 + abs(u_min))**2)

                        if q_wins:
                            x_grad[pos, score_lut.a_arr[t, r]] += v
                            x_grad[pos, score_lut.b_arr[t, r]] -= v
                        else:
                            x_grad[pos1, score_lut.a_arr[t, r]] += v
                            x_grad[pos1, score_lut.b_arr[t, r]] -= v

                        if abs_uPE < abs_uQ and abs_uPE < abs_uK:
                            rPE = int(head._pe_r_min[pe_pos, t])
                            jbarPE = j_bin ^ (1 << rPE)
                            giPE = (float(score_lut.S[t, jbarPE, 0]) - float(score_lut.S[t, j_bin, 0])) * sg
                            u_pe = float(head._pe_u_min[pe_pos, t])
                            sign_pe = 1.0 if u_pe > 0 else -1.0
                            deltaPE = giPE * (-0.5 * sign_pe / (1.0 + abs(u_pe))**2)
                            pos_grad[pe_pos, t, rPE] += deltaPE

                        if not skip_s_update:
                            score_lut.S[t, j_bin, 0] -= learning_rate * sg

            if score_grad_out is not None:
                score_grad_out[:] = score_grad

            if pe_grad_out is not None:
                pe_grad_out += pos_grad
            else:
                head.Positional_encoding -= learning_rate * pos_grad
        return

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
        if skip_s_update:
            _attention_backward_kernel_noupdate(lut.S, all_jV, all_rV, all_uV,
                                                all_jPE, all_rPE, all_uPE,
                                                y_grad, x_grad, pos_grad,
                                                lut.a_arr, lut.b_arr,
                                                args.shift_qk, POSITIONAL_DIM,
                                                CS, N_T, y_dim)
        else:
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

            if not skip_s_update:
                np.subtract.at(lut.S, (trees_b.ravel(), j_bins.ravel()),
                                np.float32(learning_rate) * y_g)

    if pe_grad_out is not None:
        pe_grad_out += pos_grad
    else:
        head.Positional_encoding -= learning_rate * pos_grad


def model_forward(m, args, training=True):
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    use_ln = getattr(args, 'layernorm', False)
    dr     = getattr(args, 'dropout', 0.0)

    for l in range(NUM_LAYERS):
        # Attention (pre-norm)
        if use_ln:
            m._x_buf[:], m._ln_attn_xhat[l][:], m._ln_attn_rstd[l][:] = \
                layernorm_forward(m.z, m.ln_attn_gamma[l], m.ln_attn_beta[l])
        else:
            np.copyto(m._x_buf, m.z)

        z_pre_attn = m.z.copy() if (dr > 0.0 and training) else None
        for h in range(NUM_HEADS):
            attention_forward(m.head[l][h], m._x_buf, m.z, args)
        if dr > 0.0 and training:
            mask = (np.random.rand(*m.z.shape) >= dr).astype(np.float32) / (1.0 - dr)
            m._drop_attn_mask[l] = mask
            m.z[:] = z_pre_attn + (m.z - z_pre_attn) * mask

        # FFN (pre-norm)
        lut = m.FFN[l]
        if use_ln:
            m._ln_ffn_buf[:], m._ln_ffn_xhat[l][:], m._ln_ffn_rstd[l][:] = \
                layernorm_forward(m.z, m.ln_ffn_gamma[l], m.ln_ffn_beta[l])
            ffn_input = m._ln_ffn_buf
        else:
            ffn_input = m.z

        z_pre_ffn = m.z.copy() if (dr > 0.0 and training) else None
        cache_index_batch(lut, ffn_input, m._ffn_j[l], m._ffn_r_min[l], m._ffn_u_min[l])
        if HAS_NUMBA and lut.S.dtype == np.float32:
            _lut_forward_add_kernel(lut.S, lut.trees, m._ffn_j[l], m.z,
                                    args.context_size, args.n_t, lut.y_dim)
        else:
            m.z[:, :lut.y_dim] += _f32(lut.S[lut.trees, m._ffn_j[l]]).sum(axis=1)
        if dr > 0.0 and training:
            mask = (np.random.rand(*m.z.shape) >= dr).astype(np.float32) / (1.0 - dr)
            m._drop_ffn_mask[l] = mask
            m.z[:] = z_pre_ffn + (m.z - z_pre_ffn) * mask

    m.output_head.forward(m.z, args)


def model_backward(m, args, grad_accum=None):
    global learning_rate
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    skip_s = grad_accum is not None
    use_ln = getattr(args, 'layernorm', False)
    dr     = getattr(args, 'dropout', 0.0)

    x_grad = np.zeros((args.context_size, EMBEDDING_DIM), dtype=np.float32)

    m.output_head.backward(x_grad, args, skip_s_update=skip_s)

    for l in range(NUM_LAYERS - 1, -1, -1):
        # FFN backward
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        if dr > 0.0 and m._drop_ffn_mask[l] is not None:
            y_grad *= m._drop_ffn_mask[l]

        lut = m.FFN[l]
        trees = lut.trees
        CS = args.context_size
        y_dim = lut.y_dim
        j = m._ffn_j[l]
        r_min = m._ffn_r_min[l]
        u_min = m._ffn_u_min[l]

        x_before_ffn = x_grad.copy() if use_ln else None

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

        if use_ln:
            ln_ffn_dx = x_grad - x_before_ffn
            dx, dgamma, dbeta = layernorm_backward(
                ln_ffn_dx, m._ln_ffn_xhat[l], m._ln_ffn_rstd[l], m.ln_ffn_gamma[l])
            x_grad[:] = x_before_ffn + dx
            if grad_accum is not None:
                grad_accum.ln_ffn_dgamma[l] += dgamma
                grad_accum.ln_ffn_dbeta[l]  += dbeta
            else:
                m.ln_ffn_gamma[l] -= learning_rate * dgamma
                m.ln_ffn_beta[l]  -= learning_rate * dbeta

        if grad_accum is not None:
            # Save y_grad for sparse replay at apply time
            grad_accum.ffn_y_grads[l] = y_grad[:, :y_dim].copy()
        else:
            if HAS_NUMBA and lut.S.dtype == np.float32:
                _lut_s_replay_kernel(lut.S, trees, j, y_grad, y_dim, CS, args.n_t,
                                     np.float32(learning_rate))
            else:
                for pos in range(CS):
                    lut.S[trees, j[pos]] -= learning_rate * y_grad[pos, :y_dim]

        # Attention backward
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        if dr > 0.0 and m._drop_attn_mask[l] is not None:
            y_grad *= m._drop_attn_mask[l]

        x_before_attn = x_grad.copy() if use_ln else None

        soft = getattr(args, 'soft_attention', False)
        if grad_accum is not None and not soft:
            # Save y_grad once per layer (shared across heads) for hard attention replay
            grad_accum.attn_y_grads[l] = y_grad.copy()
        for h in range(NUM_HEADS):
            if grad_accum is not None:
                attention_backward(m.head[l][h], x_grad, y_grad, args,
                                   skip_s_update=True,
                                   pe_grad_out=grad_accum.pe_grads[l][h],
                                   score_grad_out=grad_accum.attn_score_grads[l][h] if soft else None,
                                   val_grad_out=grad_accum.attn_val_grads[l][h] if soft else None)
            else:
                attention_backward(m.head[l][h], x_grad, y_grad, args)

        if use_ln:
            ln_attn_dx = x_grad - x_before_attn
            dx, dgamma, dbeta = layernorm_backward(
                ln_attn_dx, m._ln_attn_xhat[l], m._ln_attn_rstd[l], m.ln_attn_gamma[l])
            x_grad[:] = x_before_attn + dx
            if grad_accum is not None:
                grad_accum.ln_attn_dgamma[l] += dgamma
                grad_accum.ln_attn_dbeta[l]  += dbeta
            else:
                m.ln_attn_gamma[l] -= learning_rate * dgamma
                m.ln_attn_beta[l]  -= learning_rate * dbeta

    # no need to compute gradients for the embedder; just update the synaptic values
    # (disabled in the C version too)
    # for pos in range(args.context_size):
    #     for k in range(EMBEDDING_DIM):
    #         m.Token_embedder[m.tokens[pos]][k] -= learning_rate * x_grad[pos][k]


def model_training_step(m, args):
    model_forward(m, args, training=True)
    loss = m.output_head.training_loss(args)
    m.output_head.compute_gradients(args)
    model_backward(m, args)
    return loss


def _batch_element_step(be, args):
    """Forward + compute_gradients + backward with gradient accumulation for one batch element."""
    model_forward(be, args, training=True)
    loss = be.output_head.training_loss(args)
    be.output_head.compute_gradients(args)
    model_backward(be, args, grad_accum=be.grad_accum)
    return loss


def _replay_ffn_task(S, trees, j_list, ygrad_list, y_dim, CS, N_T, scale, use_numba):
    """Replay FFN S updates for one layer across all batch elements."""
    for j, y_grad in zip(j_list, ygrad_list):
        if use_numba and S.dtype == np.float32:
            _lut_s_replay_kernel(S, trees, j, y_grad, y_dim, CS, N_T, scale)
        else:
            for pos in range(CS):
                S[trees, j[pos]] -= scale * y_grad[pos]


def _replay_attn_task(S, jV_list, jPE_list, ygrad_list, PE, pe_grad_list,
                      shift_qk, pd, CS, N_T, y_dim, scale, use_numba):
    """Replay attention S + PE updates for one head across all batch elements."""
    for jV, jPE, y_grad in zip(jV_list, jPE_list, ygrad_list):
        if use_numba and S.dtype == np.float32:
            _attention_s_replay_kernel(S, jV, jPE, y_grad,
                                       shift_qk, pd, CS, N_T, y_dim, scale)
        else:
            trees = np.arange(N_T, dtype=np.int32)
            trees_b_full = np.broadcast_to(trees, (CS, N_T))
            for pos in range(1, CS):
                y_g = y_grad[pos, :y_dim]
                jQ = jV[pos]
                jK = jV[:pos]
                pe_idx = pos - np.arange(pos)
                jPE_slice = jPE[pe_idx]
                j_bins = (jQ << shift_qk) | (jK << pd) | jPE_slice
                trees_b = trees_b_full[:pos]
                np.subtract.at(S, (trees_b.ravel(), j_bins.ravel()), scale * y_g)
    for pe_grad in pe_grad_list:
        PE -= scale * pe_grad


def _replay_output_task(S, trees, j_list, output_list, y_dim, CS, N_T, scale, use_numba):
    """Replay output head S updates for one LUT across all batch elements."""
    for j, output in zip(j_list, output_list):
        if use_numba and S.dtype == np.float32:
            _lut_s_replay_kernel(S, trees, j, output, y_dim, CS, N_T, scale)
        else:
            for pos in range(CS):
                S[trees, j[pos]] -= scale * output[pos, :y_dim]


def _replay_soft_attn_task(ScoreS, ValueS, score_trees, val_trees,
                            jV_list, jPE_list, val_j_list, score_grad_list, val_grad_list,
                            PE, pe_grad_list, shift_qk, pd, CS, N_T, scale, use_numba=False):
    """Replay soft attention Score + Value + PE updates across all batch elements."""
    ED = ValueS.shape[2]
    if use_numba:
        for jV, jPE, val_j, sg, vg in zip(jV_list, jPE_list, val_j_list, score_grad_list, val_grad_list):
            _soft_attn_value_s_replay_kernel(ValueS, val_j, vg, CS, N_T, ED, scale)
            _soft_attn_score_s_replay_kernel(ScoreS, jV, jPE, sg, shift_qk, pd, CS, N_T, scale)
    else:
        for jV, jPE, val_j, sg, vg in zip(jV_list, jPE_list, val_j_list, score_grad_list, val_grad_list):
            # Value LUT update: each tree gets the full val_grad
            for pos1 in range(CS):
                ValueS[val_trees, val_j[pos1]] -= scale * vg[pos1]
            # Score LUT update (sequential to handle repeated j_bin collisions)
            for pos in range(1, CS):
                for pos1 in range(pos):
                    pe_pos = pos - pos1
                    scalar_sg = sg[pos, pos1]
                    for t in range(N_T):
                        j_bin = (int(jV[pos, t]) << shift_qk) | (int(jV[pos1, t]) << pd) | int(jPE[pe_pos, t])
                        ScoreS[t, j_bin, 0] -= scale * scalar_sg
    for pe_grad in pe_grad_list:
        PE -= scale * pe_grad


def apply_averaged_gradients(m, batch_elements, lr, args, pool=None):
    """Replay sparse S updates using saved y_grads and cached indices from batch elements.

    When pool is provided, replays are parallelized across independent S tables
    (6 FFN + 24 attention + 1-2 output = ~32 work units).
    """
    bs = len(batch_elements)
    scale = np.float32(lr / bs)
    CS = args.context_size
    N_T = args.n_t
    use_numba = HAS_NUMBA

    # Build work items for all independent S tables
    work = []

    for l in range(args.num_layers):
        # FFN: one task per layer
        lut = m.FFN[l]
        j_list = [be._ffn_j[l] for be in batch_elements]
        ygrad_list = [be.grad_accum.ffn_y_grads[l] for be in batch_elements]
        work.append(lambda _l=lut, _j=j_list, _y=ygrad_list:
                    _replay_ffn_task(_l.S, _l.trees, _j, _y, _l.y_dim,
                                     CS, N_T, scale, use_numba))

        # Attention: one task per (layer, head)
        for h in range(args.num_heads):
            head = m.head[l][h]
            jV_list  = [be.head[l][h]._v_j  for be in batch_elements]
            jPE_list = [be.head[l][h]._pe_j for be in batch_elements]
            pe_grad_list = [be.grad_accum.pe_grads[l][h] for be in batch_elements]
            if getattr(args, 'soft_attention', False):
                val_j_list      = [be.head[l][h]._val_j             for be in batch_elements]
                score_grad_list = [be.grad_accum.attn_score_grads[l][h] for be in batch_elements]
                val_grad_list   = [be.grad_accum.attn_val_grads[l][h]   for be in batch_elements]
                work.append(lambda _SS=head.Score.S, _VS=head.Value.S,
                                   _st=head.Score.trees, _vt=head.Value.trees,
                                   _jV=jV_list, _jPE=jPE_list, _vj=val_j_list,
                                   _sg=score_grad_list, _vg=val_grad_list,
                                   _PE=head.Positional_encoding, _pg=pe_grad_list:
                            _replay_soft_attn_task(_SS, _VS, _st, _vt,
                                                   _jV, _jPE, _vj, _sg, _vg,
                                                   _PE, _pg,
                                                   args.shift_qk, args.positional_dim,
                                                   CS, N_T, scale, use_numba))
            else:
                lut_v = head.V
                ygrad_list = [be.grad_accum.attn_y_grads[l] for be in batch_elements]
                work.append(lambda _S=lut_v.S, _jV=jV_list, _jPE=jPE_list,
                                   _y=ygrad_list, _PE=head.Positional_encoding,
                                   _pg=pe_grad_list, _yd=lut_v.y_dim:
                            _replay_attn_task(_S, _jV, _jPE, _y, _PE, _pg,
                                              args.shift_qk, args.positional_dim,
                                              CS, N_T, _yd, scale, use_numba))

    # Output head: one task per LUT
    if args.factored_output:
        for lut, j_attr, out_attr in [(m.output_head.unembedder_hi, '_hi_j', 'output_hi'),
                                      (m.output_head.unembedder_lo, '_lo_j', 'output_lo')]:
            j_list = [getattr(be.output_head, j_attr) for be in batch_elements]
            out_list = [getattr(be.output_head, out_attr) for be in batch_elements]
            work.append(lambda _l=lut, _j=j_list, _o=out_list:
                        _replay_output_task(_l.S, _l.trees, _j, _o, _l.y_dim,
                                            CS, N_T, scale, use_numba))
    else:
        lut = m.output_head.unembedder
        j_list = [be.output_head._j for be in batch_elements]
        out_list = [be.output_head.output for be in batch_elements]
        work.append(lambda _l=lut, _j=j_list, _o=out_list:
                    _replay_output_task(_l.S, _l.trees, _j, _o, _l.y_dim,
                                        CS, N_T, scale, use_numba))

    # Dispatch: parallel if pool provided, sequential otherwise
    if pool is not None:
        pool.map(lambda f: f(), work)
    else:
        for f in work:
            f()

    # LayerNorm parameter updates (averaged over batch)
    if getattr(args, 'layernorm', False):
        ln_scale = lr / len(batch_elements)
        for l in range(args.num_layers):
            m.ln_attn_gamma[l] -= ln_scale * sum(be.grad_accum.ln_attn_dgamma[l] for be in batch_elements)
            m.ln_attn_beta[l]  -= ln_scale * sum(be.grad_accum.ln_attn_dbeta[l]  for be in batch_elements)
            m.ln_ffn_gamma[l]  -= ln_scale * sum(be.grad_accum.ln_ffn_dgamma[l]  for be in batch_elements)
            m.ln_ffn_beta[l]   -= ln_scale * sum(be.grad_accum.ln_ffn_dbeta[l]   for be in batch_elements)


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
    model_forward(m, args, training=False)
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
            if getattr(args, 'soft_attention', False):
                count_array(head.Score.S)
                count_array(head.Value.S)
            else:
                count_array(head.V.S)
            count_array(head.Positional_encoding)

    # Output head
    if args.factored_output:
        count_array(m.output_head.unembedder_hi.S)
        count_array(m.output_head.unembedder_lo.S)
    else:
        count_array(m.output_head.unembedder.S)

    # LayerNorm parameters
    if getattr(args, 'layernorm', False):
        for l in range(args.num_layers):
            count_array(m.ln_attn_gamma[l]); count_array(m.ln_attn_beta[l])
            count_array(m.ln_ffn_gamma[l]);  count_array(m.ln_ffn_beta[l])

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
                        help='tiktoken encoding name, or "word" for word-based tokenizer (default: gpt2)')
    parser.add_argument('--vocab-file', type=str, default='word_vocab.txt',
                        help='Vocabulary file for --tokenizer word (built by build_word_vocab.py)')
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
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of sequences per training step (default: 1)')
    parser.add_argument('--soft-attention', action='store_true',
                        help='Decomposed Score+Value LUTs with softmax weighting (transformer-style)')
    parser.add_argument('--layernorm', action='store_true',
                        help='Pre-norm LayerNorm before attention and FFN (off by default)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate applied to residual deltas (0.0 = disabled)')
    args = parser.parse_args()

    # Initialize tokenizer and auto-set vocab size
    if args.tokenizer == 'word':
        enc = WordTokenizer(args.vocab_file)
    else:
        enc = tiktoken.get_encoding(args.tokenizer)
    args.vocab_size = enc.n_vocab
    args.enc = enc

    # Factored output dimensions
    args.vocab_hi = (args.vocab_size + 255) // 256
    args.vocab_lo = 256

    # Pre-compute positional encoding bitmasks and shift constants
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim

    # Print hyperparameters
    print("=" * 50)
    print("Hyperparameters:")
    print(f"  context_size      = {args.context_size}")
    print(f"  vocab_size        = {args.vocab_size}")
    print(f"  embedding_dim     = {args.embedding_dim}")
    print(f"  positional_dim    = {args.positional_dim}")
    print(f"  num_layers        = {args.num_layers}")
    print(f"  num_heads         = {args.num_heads}")
    print(f"  n_t               = {args.n_t}")
    print(f"  n_c               = {args.n_c}")
    print(f"  temperature       = {args.temperature}")
    print(f"  factored_output   = {args.factored_output}")
    print(f"  fp16              = {args.fp16}")
    print(f"  batch_size        = {args.batch_size}")
    print(f"  soft_attention    = {args.soft_attention}")
    print(f"  layernorm         = {args.layernorm}")
    print(f"  dropout           = {args.dropout}")
    print(f"  tokenizer         = {args.tokenizer}")
    print(f"  max_steps         = {args.max_steps}")
    print(f"  validation_interval = {args.validation_interval}")
    print(f"  testing_length    = {args.testing_length}")
    print(f"  training_data     = {args.training_data}")
    print(f"  validation_data   = {args.validation_data}")
    print(f"  loss_file         = {args.loss_file}")
    print("=" * 50)

    # Initialize loss file with header
    with open(args.loss_file, 'w') as f:
        f.write("step, loss, perplexity\n")

    training = TrainingData()
    load_training_data(training, args)

    m = Model(args)
    build_Model(m, args)
    print_model_stats(m, args)

    batch_size = args.batch_size
    if batch_size > 1:
        batch_elements = [BatchElement(m, args) for _ in range(batch_size)]
        pool = ThreadPool(batch_size)
        # Replay pool: one thread per independent S table
        n_replay = args.num_layers * (1 + args.num_heads) + (2 if args.factored_output else 1)
        replay_pool = ThreadPool(n_replay)
    else:
        batch_elements = None
        pool = None
        replay_pool = None

    pbar = tqdm(range(args.max_steps), desc="Training", unit="step")
    last_ppl = None
    last_loss = None
    ema_loss = None
    ema_alpha = 0.02

    for t in pbar:
        # Adam learning rate scheduler
        learning_rate = min(1.0 / math.sqrt(1 + t), t / 4000.0 / math.sqrt(4000))

        if batch_size == 1:
            # Fast path: identical to original code (no overhead)
            load_snippet(m, training.data, get_random_training_index(training), args)
            loss = model_training_step(m, args)
            ema_loss = loss if ema_loss is None else ema_alpha * loss + (1 - ema_alpha) * ema_loss
        else:
            # Batched path: parallel forward+backward, then averaged S update
            for b in range(batch_size):
                idx = get_random_training_index(training)
                load_snippet(batch_elements[b], training.data, idx, args)
                batch_elements[b].grad_accum.zero()

            losses = pool.map(lambda be: _batch_element_step(be, args), batch_elements)
            apply_averaged_gradients(m, batch_elements, learning_rate, args, pool=replay_pool)
            loss = sum(losses) / batch_size
            ema_loss = loss if ema_loss is None else ema_alpha * loss + (1 - ema_alpha) * ema_loss

        if t % args.validation_interval == 0:
            pbar.set_description("Validating")

            validation_loss = 0.0
            for i in tqdm(range(args.testing_length), desc="  Val", leave=False, unit="snip"):
                load_snippet(m, training.val_data, int(training.testing_input_data[i]), args)
                model_forward(m, args, training=False)
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

        train_ppl_str = f"{math.exp(min(ema_loss, 20.0)):.1f}" if ema_loss is not None else "-"
        if last_ppl is not None:
            pbar.set_postfix(train_ppl=train_ppl_str, val_ppl=f"{last_ppl:.2f}",
                             loss=f"{last_loss:.3f}", lr=f"{learning_rate:.4f}")
        elif ema_loss is not None:
            pbar.set_postfix(train_ppl=train_ppl_str, lr=f"{learning_rate:.4f}")

    if pool is not None:
        pool.close()
        pool.join()
    if replay_pool is not None:
        replay_pool.close()
        replay_pool.join()


if __name__ == '__main__':
    main()
