"""Tests comparing vectorized LUT operations against original scalar implementations."""

import numpy as np
import pytest
from types import SimpleNamespace

from main import (
    LUT, build_LUT, CONCATENATE_vec, _f32,
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

def make_args(n_t=8, n_c=4, embedding_dim=16, positional_dim=4):
    return SimpleNamespace(n_t=n_t, n_c=n_c, embedding_dim=embedding_dim,
                           positional_dim=positional_dim,
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
