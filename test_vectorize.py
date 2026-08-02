"""Tests comparing vectorized LUT operations against original scalar implementations."""

import numpy as np
import pytest
from types import SimpleNamespace

from main import (
    LUT, Model, BatchElement, build_LUT, build_Model, CONCATENATE_vec, _f32,
    apply_averaged_gradients, attention_backward, attention_forward,
    model_backward, model_forward,
    new_attention_backward, new_attention_forward,
    initialize_positional_encoding,
    build_no_overlap_comparison_arrays, build_overlap_comparison_arrays,
    configure_lut_comparison_mode,
    fill_vector_with_random_integers,
    fill_vector_with_random_integers_different_from_vector2,
)
import main as main_module


# ---------------------------------------------------------------------------
# Functions moved from main.py (used only by tests)
# ---------------------------------------------------------------------------

class LUTcache:
    __slots__ = ['r_min', 'u_min', 'j']
    def __init__(self, n_t):
        self.r_min = np.zeros(n_t, dtype=np.int32)
        self.u_min = np.zeros(n_t, dtype=np.float32)
        self.j = np.zeros(n_t, dtype=np.int32)


def cache_index(lut, cache, x, args):
    u = x[lut.a_arr] - x[lut.b_arr]
    cache.j[:] = ((u > 0).astype(np.int32) * lut.bitmasks).sum(axis=1)
    abs_u = np.abs(u)
    cache.r_min[:] = abs_u.argmin(axis=1)
    cache.u_min[:] = u[lut.trees, cache.r_min]


def cache_PE_index(cache, u, args):
    cache.j[:] = ((u > 0).astype(np.int32) * args.pe_bitmasks).sum(axis=1)
    abs_u = np.abs(u)
    cache.r_min[:] = abs_u.argmin(axis=1)
    cache.u_min[:] = u[np.arange(u.shape[0]), cache.r_min]


def LUT_forward(lut, cache, y, args):
    y[:lut.y_dim] += _f32(lut.S[lut.trees, cache.j]).sum(axis=0)


def LUT_backward(lut, cache, x_gradient, y_gradient, args):
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
    lut.S[trees, j_bins] -= main_module.learning_rate * y_gradient[:lut.y_dim]


def concatenated_LUT_forward(lut, cacheQ, cacheK, cachePE, y, args):
    j_bins = CONCATENATE_vec(cacheQ.j, cacheK.j, cachePE.j, args)
    y[:lut.y_dim] += _f32(lut.S[lut.trees, j_bins]).sum(axis=0)


def concatenated_LUT_backward(lut, cacheQ, cacheK, cachePE,
                                x_gradientQ, x_gradientK, PE_grad, y_gradient, args):
    trees = lut.trees
    y_g = y_gradient[:lut.y_dim]
    jQ, jK, jPE = cacheQ.j, cacheK.j, cachePE.j
    j_bins = CONCATENATE_vec(jQ, jK, jPE, args)
    S_j = _f32(lut.S[trees, j_bins])
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
    lut.S[trees, j_bins] -= main_module.learning_rate * y_g


# ---------------------------------------------------------------------------
# Original (scalar) reference implementations
# ---------------------------------------------------------------------------

class OrigAnchors:
    __slots__ = ['a', 'b']
    def __init__(self, n_c):
        self.a = np.zeros(n_c, dtype=np.int32)
        self.b = np.zeros(n_c, dtype=np.int32)


def orig_cache_index(anchors_list, cache, x, args):
    N_T = args.n_t
    N_C = args.n_c
    for i in range(N_T):
        cache.j[i] = 0
        cache.u_min[i] = float('inf')
        for r in range(N_C):
            u = x[anchors_list[i].a[r]] - x[anchors_list[i].b[r]]
            if u > 0:
                cache.j[i] |= (1 << r)
            if abs(u) < abs(cache.u_min[i]):
                cache.r_min[i] = r
                cache.u_min[i] = u


def orig_cache_PE_index(cache, u, args):
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim
    for i in range(N_T):
        cache.j[i] = 0
        cache.u_min[i] = float('inf')
        for r in range(POSITIONAL_DIM):
            if u[i][r] > 0:
                cache.j[i] |= (1 << r)
            if abs(u[i][r]) < abs(cache.u_min[i]):
                cache.r_min[i] = r
                cache.u_min[i] = u[i][r]


def orig_LUT_forward(S_list, y_dim, cache, y, args):
    """S_list is list of flat arrays (old format)."""
    N_T = args.n_t
    for i in range(N_T):
        offset = int(cache.j[i]) * y_dim
        y[:y_dim] += S_list[i][offset:offset + y_dim]


def orig_Up(x):
    sign_x = 1 if x > 0 else -1
    return -0.5 * sign_x / (1 + abs(x)) / (1 + abs(x))


def orig_backward_update(S_list, y_dim, anchors_list, cache, i, j, jbar, y_gradient, gradient):
    gi = 0.0
    for k in range(y_dim):
        gi += y_gradient[k] * (S_list[i][jbar + k] - S_list[i][j + k])
    v = gi * orig_Up(cache.u_min[i])
    gradient[anchors_list[i].a[cache.r_min[i]]] += v
    gradient[anchors_list[i].b[cache.r_min[i]]] -= v


def orig_LUT_backward(S_list, y_dim, anchors_list, cache, x_gradient, y_gradient, lr, args):
    N_T = args.n_t
    for i in range(N_T):
        j = int(cache.j[i]) * y_dim
        jbar = int(cache.j[i] ^ (1 << cache.r_min[i])) * y_dim
        orig_backward_update(S_list, y_dim, anchors_list, cache, i, j, jbar, y_gradient, x_gradient)
        S_list[i][j:j + y_dim] -= lr * y_gradient[:y_dim]


def orig_CONCATENATE(Q, P, PE, y_dim, args):
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    return int(((Q << (N_C + POSITIONAL_DIM)) | (P << POSITIONAL_DIM) | PE) * y_dim)


def orig_concatenated_LUT_forward(S_list, y_dim, cacheQ, cacheK, cachePE, y, args):
    N_T = args.n_t
    for i in range(N_T):
        j = orig_CONCATENATE(int(cacheQ.j[i]), int(cacheK.j[i]), int(cachePE.j[i]), y_dim, args)
        y[:y_dim] += S_list[i][j:j + y_dim]


def orig_concatenated_LUT_backward(S_list, y_dim, anchors_list, cacheQ, cacheK, cachePE,
                                    x_gradientQ, x_gradientK, PE_grad, y_gradient, lr, args):
    N_T = args.n_t
    for i in range(N_T):
        j = orig_CONCATENATE(int(cacheQ.j[i]), int(cacheK.j[i]), int(cachePE.j[i]), y_dim, args)

        if abs(cacheQ.u_min[i]) < abs(cacheK.u_min[i]):
            jbar = orig_CONCATENATE(
                int(cacheQ.j[i] ^ (1 << cacheQ.r_min[i])),
                int(cacheK.j[i]),
                int(cachePE.j[i]),
                y_dim, args)
            orig_backward_update(S_list, y_dim, anchors_list, cacheQ, i, j, jbar, y_gradient, x_gradientQ)
        else:
            jbar = orig_CONCATENATE(
                int(cacheQ.j[i]),
                int(cacheK.j[i] ^ (1 << cacheK.r_min[i])),
                int(cachePE.j[i]),
                y_dim, args)
            orig_backward_update(S_list, y_dim, anchors_list, cacheK, i, j, jbar, y_gradient, x_gradientK)

        if (abs(cachePE.u_min[i]) < abs(cacheQ.u_min[i]) and
                abs(cachePE.u_min[i]) < abs(cacheK.u_min[i])):
            jbarPE = orig_CONCATENATE(
                int(cacheQ.j[i]),
                int(cacheK.j[i]),
                int(cachePE.j[i] ^ (1 << cachePE.r_min[i])),
                y_dim, args)
            giPE = 0.0
            for k in range(y_dim):
                giPE += y_gradient[k] * (S_list[i][jbarPE + k] - S_list[i][j + k])
            deltaPE = giPE * orig_Up(cachePE.u_min[i])
            PE_grad[i][cachePE.r_min[i]] += deltaPE

        S_list[i][j:j + y_dim] -= lr * y_gradient[:y_dim]


# ---------------------------------------------------------------------------
# Helper: create matched old/new format data from same random seed
# ---------------------------------------------------------------------------

def make_args(n_t=8, n_c=4, embedding_dim=16, positional_dim=4,
              lut_comparison_mode='random'):
    return SimpleNamespace(n_t=n_t, n_c=n_c, embedding_dim=embedding_dim,
                           positional_dim=positional_dim,
                           freeze_PE=False,
                           lut_comparison_mode=lut_comparison_mode,
                           pe_bitmasks=(1 << np.arange(positional_dim)).astype(np.int32),
                           shift_qk=n_c + positional_dim)


def make_matched_lut_and_old_data(args, total_n_c=None, y_dim=8, seed=42):
    """Build a new-format LUT and matching old-format data from the same seed."""
    import random as py_random

    if total_n_c is None:
        total_n_c = args.n_c

    N_T = args.n_t
    N_C = args.n_c
    EMBEDDING_DIM = args.embedding_dim
    num_bins = 1 << total_n_c

    # Build new-format LUT
    py_random.seed(seed)
    np.random.seed(seed)
    lut = LUT(N_T)
    build_LUT(lut, total_n_c, y_dim, args)

    # Build old-format data from same seed
    py_random.seed(seed)
    np.random.seed(seed)
    anchors_list = []
    S_list = []
    for i in range(N_T):
        anc = OrigAnchors(N_C)
        anc.a = fill_vector_with_random_integers(N_C, EMBEDDING_DIM)
        anc.b = fill_vector_with_random_integers_different_from_vector2(anc.a, N_C, EMBEDDING_DIM)
        anchors_list.append(anc)
        S_list.append(np.zeros(num_bins * y_dim, dtype=np.float32))

    # Verify anchors match
    for i in range(N_T):
        assert np.array_equal(lut.a_arr[i], anchors_list[i].a)
        assert np.array_equal(lut.b_arr[i], anchors_list[i].b)

    # Fill S with random data (same for both)
    np.random.seed(seed + 100)
    for i in range(N_T):
        data = np.random.randn(num_bins * y_dim).astype(np.float32)
        S_list[i][:] = data
        lut.S[i] = data.reshape(num_bins, y_dim)

    return lut, anchors_list, S_list, y_dim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ParserStub:
    def error(self, message):
        raise ValueError(message)


class TestLUTComparisonModes:
    def test_no_overlap_dim64_nc6_uses_16_four_neuron_tables(self):
        a_arr, b_arr = build_no_overlap_comparison_arrays(6, 64)

        assert a_arr.shape == (16, 6)
        assert b_arr.shape == (16, 6)
        np.testing.assert_array_equal(a_arr[0], np.array([0, 0, 0, 1, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(b_arr[0], np.array([1, 2, 3, 2, 3, 3], dtype=np.int32))
        np.testing.assert_array_equal(a_arr[-1], np.array([60, 60, 60, 61, 61, 62], dtype=np.int32))
        np.testing.assert_array_equal(b_arr[-1], np.array([61, 62, 63, 62, 63, 63], dtype=np.int32))

    def test_no_overlap_pads_final_group_when_dim_is_not_divisible(self):
        a_arr, b_arr = build_no_overlap_comparison_arrays(6, 10)

        assert a_arr.shape == (3, 6)
        np.testing.assert_array_equal(a_arr[-1], np.array([8, 8, 8, 9, 9, 0], dtype=np.int32))
        np.testing.assert_array_equal(b_arr[-1], np.array([9, 0, 1, 0, 1, 1], dtype=np.int32))

    def test_overlap_keeps_requested_n_t_and_adds_pairwise_half_tables(self):
        a_arr, b_arr = build_overlap_comparison_arrays(18, 6, 64)

        assert a_arr.shape == (18, 6)
        np.testing.assert_array_equal(a_arr[0], np.array([0, 0, 0, 1, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(b_arr[0], np.array([1, 2, 3, 2, 3, 3], dtype=np.int32))
        np.testing.assert_array_equal(a_arr[16], np.array([0, 0, 0, 1, 1, 6], dtype=np.int32))
        np.testing.assert_array_equal(b_arr[16], np.array([1, 6, 7, 6, 7, 7], dtype=np.int32))
        np.testing.assert_array_equal(a_arr[17], np.array([4, 4, 4, 5, 5, 10], dtype=np.int32))
        np.testing.assert_array_equal(b_arr[17], np.array([5, 10, 11, 10, 11, 11], dtype=np.int32))

    def test_random_mode_remains_seed_compatible(self):
        import random as py_random

        args = make_args(n_t=3, n_c=4, embedding_dim=16, lut_comparison_mode='random')
        py_random.seed(7)
        lut = LUT(args.n_t)
        build_LUT(lut, args.n_c, 2, args)

        py_random.seed(7)
        expected_a = np.zeros((args.n_t, args.n_c), dtype=np.int32)
        expected_b = np.zeros((args.n_t, args.n_c), dtype=np.int32)
        for i in range(args.n_t):
            expected_a[i] = fill_vector_with_random_integers(args.n_c, args.embedding_dim)
            expected_b[i] = fill_vector_with_random_integers_different_from_vector2(
                expected_a[i], args.n_c, args.embedding_dim)

        np.testing.assert_array_equal(lut.a_arr, expected_a)
        np.testing.assert_array_equal(lut.b_arr, expected_b)

    def test_build_model_shapes_use_effective_no_overlap_table_counts(self):
        args = SimpleNamespace(
            vocab_size_input=8,
            vocab_size_output=5,
            context_size=3,
            embedding_dim=10,
            positional_dim=2,
            num_layers=1,
            num_heads=1,
            n_t=99,
            n_c=6,
            lut_comparison_mode='no-overlap',
            attn_qk_dim=4,
            attn_qk_n_c=3,
            attn_qk_n_t=99,
            factored_output=False,
            fp16=False,
            batch_size=1,
            cpu_cores=1,
            num_workers=1,
            numba_threads=1,
            replay_workers=1,
            soft_attention=False,
            layernorm=False,
            dropout=0.0,
            POS_task=False,
            new_attention=True,
            PE_only=False,
            freeze_PE=False,
            qkpe_direct_sign=False,
        )
        args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
        args.shift_qk = args.n_c + args.positional_dim
        configure_lut_comparison_mode(args, ParserStub())

        model = Model(args)
        build_Model(model, args)
        head = model.head[0][0]

        assert args.n_t == 3
        assert args.attn_qk_n_t == 4
        assert model.FFN[0].S.shape[0] == args.n_t
        assert head.V.S.shape[0] == args.n_t
        assert head.Final.S.shape[0] == args.n_t
        assert head.Q.S.shape[0] == args.attn_qk_n_t
        assert head.K.S.shape[0] == args.attn_qk_n_t
        assert head._qkpe_a_arr.shape[0] == args.n_t


class TestCacheIndex:
    def test_cache_index(self):
        args = make_args()
        lut, anchors_list, _, _ = make_matched_lut_and_old_data(args)
        x = np.random.randn(args.embedding_dim).astype(np.float32)

        cache_new = LUTcache(args.n_t)
        cache_old = LUTcache(args.n_t)

        cache_index(lut, cache_new, x, args)
        orig_cache_index(anchors_list, cache_old, x, args)

        np.testing.assert_array_equal(cache_new.j, cache_old.j)
        np.testing.assert_array_equal(cache_new.r_min, cache_old.r_min)
        np.testing.assert_array_equal(cache_new.u_min, cache_old.u_min)


class TestCachePEIndex:
    def test_cache_PE_index(self):
        args = make_args()
        u = np.random.randn(args.n_t, args.positional_dim).astype(np.float32)

        cache_new = LUTcache(args.n_t)
        cache_old = LUTcache(args.n_t)

        cache_PE_index(cache_new, u, args)
        orig_cache_PE_index(cache_old, u, args)

        np.testing.assert_array_equal(cache_new.j, cache_old.j)
        np.testing.assert_array_equal(cache_new.r_min, cache_old.r_min)
        np.testing.assert_array_equal(cache_new.u_min, cache_old.u_min)


class TestLUTForward:
    def test_LUT_forward(self):
        args = make_args()
        lut, _, S_list, y_dim = make_matched_lut_and_old_data(args)

        # Create matching cache state
        cache = LUTcache(args.n_t)
        np.random.seed(77)
        num_bins = 1 << args.n_c
        cache.j[:] = np.random.randint(0, num_bins, args.n_t).astype(np.int32)

        y_new = np.zeros(y_dim, dtype=np.float32)
        y_old = np.zeros(y_dim, dtype=np.float32)

        LUT_forward(lut, cache, y_new, args)
        orig_LUT_forward(S_list, y_dim, cache, y_old, args)

        np.testing.assert_allclose(y_new, y_old, atol=1e-6)


class TestLUTBackward:
    def test_LUT_backward(self):
        args = make_args()
        lut, anchors_list, S_list, y_dim = make_matched_lut_and_old_data(args)
        lr = 0.01

        # Create matching cache state
        cache = LUTcache(args.n_t)
        np.random.seed(77)
        num_bins = 1 << args.n_c
        cache.j[:] = np.random.randint(0, num_bins, args.n_t).astype(np.int32)
        cache.r_min[:] = np.random.randint(0, args.n_c, args.n_t).astype(np.int32)
        cache.u_min[:] = np.random.randn(args.n_t).astype(np.float32)

        y_gradient = np.random.randn(y_dim).astype(np.float32)

        x_grad_new = np.zeros(args.embedding_dim, dtype=np.float32)
        x_grad_old = np.zeros(args.embedding_dim, dtype=np.float32)

        # Need separate caches since backward modifies S
        cache_new = LUTcache(args.n_t)
        cache_new.j[:] = cache.j[:]
        cache_new.r_min[:] = cache.r_min[:]
        cache_new.u_min[:] = cache.u_min[:]

        main_module.learning_rate = lr
        LUT_backward(lut, cache_new, x_grad_new, y_gradient, args)
        orig_LUT_backward(S_list, y_dim, anchors_list, cache, x_grad_old, y_gradient, lr, args)

        np.testing.assert_allclose(x_grad_new, x_grad_old, atol=1e-6)
        # Check S was updated identically
        for i in range(args.n_t):
            np.testing.assert_allclose(lut.S[i].ravel(), S_list[i], atol=1e-6)


class TestConcatenatedLUTForward:
    def test_concatenated_LUT_forward(self):
        args = make_args(n_c=3, positional_dim=3)
        total_n_c = args.n_c + args.n_c + args.positional_dim
        y_dim = 8
        lut, _, S_list, _ = make_matched_lut_and_old_data(args, total_n_c=total_n_c, y_dim=y_dim)

        np.random.seed(88)
        num_bins_nc = 1 << args.n_c
        num_bins_pd = 1 << args.positional_dim

        cacheQ = LUTcache(args.n_t)
        cacheK = LUTcache(args.n_t)
        cachePE = LUTcache(args.n_t)
        cacheQ.j[:] = np.random.randint(0, num_bins_nc, args.n_t).astype(np.int32)
        cacheK.j[:] = np.random.randint(0, num_bins_nc, args.n_t).astype(np.int32)
        cachePE.j[:] = np.random.randint(0, num_bins_pd, args.n_t).astype(np.int32)

        y_new = np.zeros(y_dim, dtype=np.float32)
        y_old = np.zeros(y_dim, dtype=np.float32)

        concatenated_LUT_forward(lut, cacheQ, cacheK, cachePE, y_new, args)
        orig_concatenated_LUT_forward(S_list, y_dim, cacheQ, cacheK, cachePE, y_old, args)

        np.testing.assert_allclose(y_new, y_old, atol=1e-6)


class TestConcatenatedLUTBackward:
    def test_concatenated_LUT_backward(self):
        args = make_args(n_c=3, positional_dim=3)
        total_n_c = args.n_c + args.n_c + args.positional_dim
        y_dim = 8
        lut, anchors_list, S_list, _ = make_matched_lut_and_old_data(
            args, total_n_c=total_n_c, y_dim=y_dim)
        lr = 0.01

        np.random.seed(88)
        num_bins_nc = 1 << args.n_c
        num_bins_pd = 1 << args.positional_dim

        # Create caches with random state
        cacheQ = LUTcache(args.n_t)
        cacheK = LUTcache(args.n_t)
        cachePE = LUTcache(args.n_t)
        cacheQ.j[:] = np.random.randint(0, num_bins_nc, args.n_t).astype(np.int32)
        cacheK.j[:] = np.random.randint(0, num_bins_nc, args.n_t).astype(np.int32)
        cachePE.j[:] = np.random.randint(0, num_bins_pd, args.n_t).astype(np.int32)
        cacheQ.r_min[:] = np.random.randint(0, args.n_c, args.n_t).astype(np.int32)
        cacheK.r_min[:] = np.random.randint(0, args.n_c, args.n_t).astype(np.int32)
        cachePE.r_min[:] = np.random.randint(0, args.positional_dim, args.n_t).astype(np.int32)
        cacheQ.u_min[:] = np.random.randn(args.n_t).astype(np.float32)
        cacheK.u_min[:] = np.random.randn(args.n_t).astype(np.float32)
        cachePE.u_min[:] = np.random.randn(args.n_t).astype(np.float32)

        y_gradient = np.random.randn(y_dim).astype(np.float32)

        # New implementation
        x_gradQ_new = np.zeros(args.embedding_dim, dtype=np.float32)
        x_gradK_new = np.zeros(args.embedding_dim, dtype=np.float32)
        PE_grad_new = np.zeros((args.n_t, args.positional_dim), dtype=np.float32)

        # Old implementation
        x_gradQ_old = np.zeros(args.embedding_dim, dtype=np.float32)
        x_gradK_old = np.zeros(args.embedding_dim, dtype=np.float32)
        PE_grad_old = np.zeros((args.n_t, args.positional_dim), dtype=np.float32)

        main_module.learning_rate = lr
        concatenated_LUT_backward(lut, cacheQ, cacheK, cachePE,
                                   x_gradQ_new, x_gradK_new, PE_grad_new, y_gradient, args)
        orig_concatenated_LUT_backward(S_list, y_dim, anchors_list, cacheQ, cacheK, cachePE,
                                        x_gradQ_old, x_gradK_old, PE_grad_old, y_gradient, lr, args)

        np.testing.assert_allclose(x_gradQ_new, x_gradQ_old, atol=1e-6)
        np.testing.assert_allclose(x_gradK_new, x_gradK_old, atol=1e-6)
        np.testing.assert_allclose(PE_grad_new, PE_grad_old, atol=1e-6)
        # Check S was updated identically
        for i in range(args.n_t):
            np.testing.assert_allclose(lut.S[i].ravel(), S_list[i], atol=1e-6)



def randomize_lut_values(lut):
    lut.S[:] = np.random.randn(*lut.S.shape).astype(np.float32)


def randomize_pe_only_attention_values(model):
    head = model.head[0][0]
    randomize_lut_values(head.Q)
    randomize_lut_values(head.K)
    randomize_lut_values(head.V)
    randomize_lut_values(head.Final)
    return head


def make_pe_only_model_args(batch_size=1, freeze_PE=False):
    args = SimpleNamespace(
        vocab_size_input=8,
        vocab_size_output=5,
        context_size=4,
        embedding_dim=4,
        positional_dim=2,
        num_layers=1,
        num_heads=1,
        n_t=2,
        n_c=2,
        attn_qk_dim=2,
        attn_qk_n_c=2,
        attn_qk_n_t=2,
        factored_output=False,
        fp16=False,
        batch_size=batch_size,
        cpu_cores=1,
        num_workers=1,
        numba_threads=1,
        replay_workers=1,
        soft_attention=False,
        layernorm=False,
        dropout=0.0,
        POS_task=False,
        new_attention=True,
        PE_only=True,
        freeze_PE=freeze_PE,
        qkpe_direct_sign=False,
        PE_init_scale=1.0,
    )
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim
    return args


class TestPEOnlyNewAttention:
    def test_forward_ignores_qk_and_uses_pe(self):
        args = make_pe_only_model_args()
        np.random.seed(123)
        model = Model(args)
        build_Model(model, args)
        head = randomize_pe_only_attention_values(model)
        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)

        y_base = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        new_attention_forward(head, x, y_base, args)
        np.testing.assert_array_equal(head._q_embed, np.zeros_like(head._q_embed))
        np.testing.assert_array_equal(head._k_embed, np.zeros_like(head._k_embed))

        head.Q.S += np.random.randn(*head.Q.S.shape).astype(np.float32) * 100.0
        head.K.S += np.random.randn(*head.K.S.shape).astype(np.float32) * 100.0
        y_after_qk = np.zeros_like(y_base)
        new_attention_forward(head, x, y_after_qk, args)
        np.testing.assert_allclose(y_after_qk, y_base, atol=1e-6)

        head.Positional_encoding[:] = 0.0
        y_pe_zero = np.zeros_like(y_base)
        new_attention_forward(head, x, y_pe_zero, args)
        head.Positional_encoding[:, 0] = 10.0
        head.Positional_encoding[:, 1] = -10.0
        y_pe_changed = np.zeros_like(y_base)
        new_attention_forward(head, x, y_pe_changed, args)
        assert not np.allclose(y_pe_changed, y_pe_zero)

    def test_backward_does_not_update_qk(self):
        args = make_pe_only_model_args()
        np.random.seed(456)
        model = Model(args)
        build_Model(model, args)
        head = randomize_pe_only_attention_values(model)
        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
        y = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        new_attention_forward(head, x, y, args)

        q_before = head.Q.S.copy()
        k_before = head.K.S.copy()
        final_before = head.Final.S.copy()
        pe_before = head.Positional_encoding.copy()
        x_grad = np.zeros_like(x)
        y_grad = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)

        main_module.learning_rate = 0.05
        new_attention_backward(head, x_grad, y_grad, args)

        np.testing.assert_array_equal(head.Q.S, q_before)
        np.testing.assert_array_equal(head.K.S, k_before)
        assert np.any(head.Final.S != final_before)
        assert np.any(head.Positional_encoding != pe_before)

    def test_freeze_pe_keeps_pe_grad_zero_and_updates_final(self):
        args = make_pe_only_model_args(freeze_PE=True)
        np.random.seed(457)
        model = Model(args)
        build_Model(model, args)
        head = randomize_pe_only_attention_values(model)
        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
        y = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        new_attention_forward(head, x, y, args)

        final_before = head.Final.S.copy()
        pe_before = head.Positional_encoding.copy()
        x_grad = np.zeros_like(x)
        y_grad = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
        pe_grad = np.zeros_like(head._pe_grad)

        main_module.learning_rate = 0.05
        new_attention_backward(head, x_grad, y_grad, args, pe_grad_out=pe_grad)

        assert np.any(head.Final.S != final_before)
        np.testing.assert_array_equal(head.Positional_encoding, pe_before)
        np.testing.assert_array_equal(pe_grad, np.zeros_like(pe_grad))

    def test_batch_replay_does_not_update_qk(self):
        args = make_pe_only_model_args(batch_size=2)
        np.random.seed(789)
        model = Model(args)
        build_Model(model, args)
        randomize_pe_only_attention_values(model)
        randomize_lut_values(model.output_head.unembedder)
        batch_elements = [BatchElement(model, args) for _ in range(args.batch_size)]

        for i, be in enumerate(batch_elements):
            be.z[:] = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
            be.tokens[:] = np.arange(args.context_size + 1, dtype=np.int32) % args.vocab_size_input
            model_forward(be, args, training=True)
            be.output_head.output[:] = (i + 1) * np.ones_like(be.output_head.output, dtype=np.float32)
            model_backward(be, args, grad_accum=be.grad_accum)

        head = model.head[0][0]
        q_before = head.Q.S.copy()
        k_before = head.K.S.copy()
        final_before = head.Final.S.copy()
        pe_before = head.Positional_encoding.copy()

        apply_averaged_gradients(model, batch_elements, 0.05, args)

        np.testing.assert_array_equal(head.Q.S, q_before)
        np.testing.assert_array_equal(head.K.S, k_before)
        assert np.any(head.Final.S != final_before)
        assert np.any(head.Positional_encoding != pe_before)

    def test_freeze_pe_batch_replay_keeps_pe_fixed(self):
        args = make_pe_only_model_args(batch_size=2, freeze_PE=True)
        np.random.seed(790)
        model = Model(args)
        build_Model(model, args)
        randomize_pe_only_attention_values(model)
        randomize_lut_values(model.output_head.unembedder)
        batch_elements = [BatchElement(model, args) for _ in range(args.batch_size)]

        for i, be in enumerate(batch_elements):
            be.z[:] = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
            be.tokens[:] = np.arange(args.context_size + 1, dtype=np.int32) % args.vocab_size_input
            model_forward(be, args, training=True)
            be.output_head.output[:] = (i + 1) * np.ones_like(be.output_head.output, dtype=np.float32)
            model_backward(be, args, grad_accum=be.grad_accum)

        head = model.head[0][0]
        final_before = head.Final.S.copy()
        pe_before = head.Positional_encoding.copy()

        apply_averaged_gradients(model, batch_elements, 0.05, args)

        assert np.any(head.Final.S != final_before)
        np.testing.assert_array_equal(head.Positional_encoding, pe_before)



def make_attention_freeze_model_args(soft_attention=False):
    args = SimpleNamespace(
        vocab_size_input=8,
        vocab_size_output=5,
        context_size=4,
        embedding_dim=4,
        positional_dim=2,
        num_layers=1,
        num_heads=1,
        n_t=2,
        n_c=2,
        attn_qk_dim=2,
        attn_qk_n_c=2,
        attn_qk_n_t=2,
        factored_output=False,
        fp16=False,
        batch_size=1,
        cpu_cores=1,
        num_workers=1,
        numba_threads=1,
        replay_workers=1,
        soft_attention=soft_attention,
        layernorm=False,
        dropout=0.0,
        POS_task=False,
        new_attention=False,
        PE_only=False,
        freeze_PE=True,
        qkpe_direct_sign=False,
    )
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim
    return args


class TestFreezePEAttention:
    def run_frozen_attention_backward(self, soft_attention):
        args = make_attention_freeze_model_args(soft_attention=soft_attention)
        np.random.seed(901 + int(soft_attention))
        model = Model(args)
        build_Model(model, args)
        head = model.head[0][0]
        if soft_attention:
            randomize_lut_values(head.Score)
            randomize_lut_values(head.Value)
        else:
            randomize_lut_values(head.V)

        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
        y = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        attention_forward(head, x, y, args)
        pe_before = head.Positional_encoding.copy()
        x_grad = np.zeros_like(x)
        y_grad = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)

        main_module.learning_rate = 0.05
        attention_backward(head, x_grad, y_grad, args)

        np.testing.assert_array_equal(head.Positional_encoding, pe_before)

    def test_hard_attention_freeze_pe_keeps_pe_fixed(self):
        self.run_frozen_attention_backward(soft_attention=False)

    def test_soft_attention_freeze_pe_keeps_pe_fixed(self):
        self.run_frozen_attention_backward(soft_attention=True)



def make_frozen_pe_init_args(new_attention=False, soft_attention=False):
    args = SimpleNamespace(
        vocab_size_input=8,
        vocab_size_output=5,
        context_size=4,
        embedding_dim=4,
        positional_dim=2,
        num_layers=2,
        num_heads=2,
        n_t=3,
        n_c=2,
        attn_qk_dim=3,
        attn_qk_n_c=2,
        attn_qk_n_t=2,
        factored_output=False,
        fp16=False,
        batch_size=1,
        cpu_cores=1,
        num_workers=1,
        numba_threads=1,
        replay_workers=1,
        soft_attention=soft_attention,
        layernorm=False,
        dropout=0.0,
        POS_task=False,
        new_attention=new_attention,
        PE_only=False,
        freeze_PE=True,
        qkpe_direct_sign=False,
        PE_init_scale=1.0,
    )
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim
    return args


class TestFreezePEInitialization:
    def assert_all_old_attention_pe_shared(self, model, args):
        template = model.head[0][0].Positional_encoding[:, 0, :]
        assert np.any(template != 0.0)
        for l in range(args.num_layers):
            for h in range(args.num_heads):
                pe = model.head[l][h].Positional_encoding
                for t in range(args.n_t):
                    np.testing.assert_array_equal(pe[:, t, :], template)


    def test_initializer_uses_binary_row_signs_and_scaled_random_magnitudes(self):
        args = SimpleNamespace(positional_dim=5, PE_init_scale=0.58)
        np.random.seed(1301)
        pe = initialize_positional_encoding((32, 5), args)

        assert pe.shape == (32, 5)
        assert np.all(np.abs(pe) <= np.float32(args.PE_init_scale))
        assert np.all(np.abs(pe) >= 0.0)

        expected_signs = np.array([
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1,  1],
            [-1, -1, -1,  1, -1],
            [ 1,  1,  1,  1,  1],
        ], dtype=np.float32)
        observed_signs = np.sign(pe[[0, 1, 2, 31]])
        np.testing.assert_array_equal(observed_signs, expected_signs)

    def test_hard_attention_freeze_pe_shares_random_template_across_tables_heads_layers(self):
        args = make_frozen_pe_init_args(new_attention=False, soft_attention=False)
        np.random.seed(1201)
        model = Model(args)
        build_Model(model, args)

        self.assert_all_old_attention_pe_shared(model, args)

    def test_soft_attention_freeze_pe_shares_random_template_across_tables_heads_layers(self):
        args = make_frozen_pe_init_args(new_attention=False, soft_attention=True)
        np.random.seed(1202)
        model = Model(args)
        build_Model(model, args)

        self.assert_all_old_attention_pe_shared(model, args)

    def test_new_attention_freeze_pe_shares_random_template_across_heads_layers(self):
        args = make_frozen_pe_init_args(new_attention=True, soft_attention=False)
        np.random.seed(1203)
        model = Model(args)
        build_Model(model, args)

        template = model.head[0][0].Positional_encoding
        assert np.any(template != 0.0)
        for l in range(args.num_layers):
            for h in range(args.num_heads):
                np.testing.assert_array_equal(
                    model.head[l][h].Positional_encoding, template)


def make_qkpe_direct_model_args(pe_only=False):
    args = SimpleNamespace(
        vocab_size_input=8,
        vocab_size_output=5,
        context_size=3,
        embedding_dim=3,
        positional_dim=2,
        num_layers=1,
        num_heads=1,
        n_t=1,
        n_c=1,
        attn_qk_dim=3,
        attn_qk_n_c=1,
        attn_qk_n_t=1,
        factored_output=False,
        fp16=False,
        batch_size=1,
        cpu_cores=1,
        num_workers=1,
        numba_threads=1,
        replay_workers=1,
        soft_attention=False,
        layernorm=False,
        dropout=0.0,
        POS_task=False,
        new_attention=True,
        PE_only=pe_only,
        freeze_PE=False,
        qkpe_direct_sign=True,
    )
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim
    return args


def expected_direct_qkpe_j(head, args, pe_only=False):
    expected = np.zeros_like(head._qkpe_j)
    for tok_i in range(1, args.context_size):
        for tok_j in range(tok_i):
            pair = tok_i * (tok_i - 1) // 2 + tok_j
            pe_pos = tok_i - tok_j
            if pe_only:
                u = head.Positional_encoding[pe_pos, :args.positional_dim]
            else:
                u = (head._q_embed[tok_i, :args.positional_dim]
                     + head._k_embed[tok_j, :args.positional_dim]
                     + head.Positional_encoding[pe_pos, :args.positional_dim])
            j = int(((u > 0).astype(np.int32) * args.pe_bitmasks).sum())
            expected[pair, :] = j
    return expected


class TestQKPEDirectSign:
    def test_forward_bins_use_qkpe_coordinate_signs_and_ignore_anchor_pairs(self):
        args = make_qkpe_direct_model_args(pe_only=False)
        np.random.seed(321)
        model = Model(args)
        build_Model(model, args)
        head = randomize_pe_only_attention_values(model)
        head.Positional_encoding[:] = np.array([
            [0.0, 0.0, 0.0],
            [0.25, -0.75, 0.5],
            [-0.5, 0.5, -0.25],
        ], dtype=np.float32)
        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)

        y = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        new_attention_forward(head, x, y, args)
        expected = expected_direct_qkpe_j(head, args, pe_only=False)
        np.testing.assert_array_equal(head._qkpe_j, expected)

        head._qkpe_a_arr[:] = 2
        head._qkpe_b_arr[:] = 1
        y_after_anchor_change = np.zeros_like(y)
        new_attention_forward(head, x, y_after_anchor_change, args)
        np.testing.assert_array_equal(head._qkpe_j, expected)

    def test_pe_only_forward_bins_use_pe_coordinate_signs(self):
        args = make_qkpe_direct_model_args(pe_only=True)
        np.random.seed(654)
        model = Model(args)
        build_Model(model, args)
        head = randomize_pe_only_attention_values(model)
        head.Positional_encoding[:] = np.array([
            [0.0, 0.0, 0.0],
            [0.5, -0.5, 1.0],
            [-0.25, 0.75, -1.0],
        ], dtype=np.float32)
        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)

        y = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        new_attention_forward(head, x, y, args)
        np.testing.assert_array_equal(head._q_embed, np.zeros_like(head._q_embed))
        np.testing.assert_array_equal(head._k_embed, np.zeros_like(head._k_embed))
        np.testing.assert_array_equal(
            head._qkpe_j, expected_direct_qkpe_j(head, args, pe_only=True))

    def test_backward_direct_scatter_uses_fragile_coordinate_not_anchor_pair(self):
        args = make_qkpe_direct_model_args(pe_only=False)
        np.random.seed(987)
        model = Model(args)
        build_Model(model, args)
        head = randomize_pe_only_attention_values(model)
        x = np.random.randn(args.context_size, args.embedding_dim).astype(np.float32)
        y = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        new_attention_forward(head, x, y, args)

        pair = 0
        tree = 0
        r_qkpe = 1
        head._qkpe_r_min[pair, tree] = r_qkpe
        head._qkpe_u_min[pair, tree] = 0.5
        head._qkpe_a_arr[tree, r_qkpe] = 2
        head._qkpe_b_arr[tree, r_qkpe] = 0

        j_bin = (int(head._final_v_j[0, tree]) << args.positional_dim) | int(head._qkpe_j[pair, tree])
        jbar_qkpe = j_bin ^ (1 << r_qkpe)
        head.Final.S[:] = 0.0
        head.Final.S[head.Final.trees[tree], jbar_qkpe, :] = 1.0

        q_grad = np.zeros((args.context_size, args.attn_qk_dim), dtype=np.float32)
        k_grad = np.zeros_like(q_grad)
        v_grad = np.zeros((args.context_size, args.embedding_dim), dtype=np.float32)
        pe_grad = np.zeros_like(q_grad)
        x_grad = np.zeros_like(x)
        y_grad = np.zeros_like(y)
        y_grad[1, :] = 1.0

        main_module.learning_rate = 0.0
        new_attention_backward(
            head, x_grad, y_grad, args, skip_s_update=True,
            q_grad_out=q_grad, k_grad_out=k_grad,
            v_grad_out=v_grad, pe_grad_out=pe_grad)

        assert q_grad[1, r_qkpe] != 0.0
        assert k_grad[0, r_qkpe] != 0.0
        assert pe_grad[1, r_qkpe] != 0.0
        np.testing.assert_array_equal(q_grad[:, 2], np.zeros(args.context_size, dtype=np.float32))
        np.testing.assert_array_equal(k_grad[:, 2], np.zeros(args.context_size, dtype=np.float32))
        np.testing.assert_array_equal(pe_grad[:, 2], np.zeros(args.context_size, dtype=np.float32))
