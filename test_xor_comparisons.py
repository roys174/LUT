from types import SimpleNamespace
import random
import sys

import numpy as np
import pytest


class TiktokenStub:
    @staticmethod
    def get_encoding(_name):
        raise RuntimeError('tiktoken is not needed by these tests')


sys.modules.setdefault('tiktoken', TiktokenStub())

import main as main_module
from main import (
    LUT,
    _lut_backward_python,
    _lut_load,
    _lut_save,
    build_LUT,
    build_comparison_arrays,
    cache_index_batch,
    configure_lut_comparison_mode,
)


class ParserStub:
    def error(self, message):
        raise ValueError(message)


def make_args(**overrides):
    args = SimpleNamespace(
        n_t=2,
        n_c=2,
        attn_qk_n_c=1,
        attn_qk_n_t=2,
        embedding_dim=8,
        lut_comparison_mode='random',
        comparisons_per_bit=1,
        fp16=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def legacy_cache_index(a_arr, b_arr, bitmasks, x):
    u = x[:, a_arr] - x[:, b_arr]
    j = ((u > 0).astype(np.int32) * bitmasks).sum(axis=2)
    r_min = np.abs(u).argmin(axis=2).astype(np.int32)
    positions = np.arange(x.shape[0])[:, None]
    tables = np.arange(a_arr.shape[0])
    u_min = u[positions, tables, r_min]
    return j, r_min, u_min


def legacy_backward(lut, j, r_min, u_min, grad, learning_rate):
    x_grad = np.zeros((j.shape[0], 8), dtype=np.float32)
    expected_s = lut.S.copy()
    for pos in range(j.shape[0]):
        for t in range(j.shape[1]):
            tree = int(lut.trees[t])
            j_bin = int(j[pos, t])
            r = int(r_min[pos, t])
            jbar = j_bin ^ (1 << r)
            gi = (
                (expected_s[tree, jbar] - expected_s[tree, j_bin])
                * grad[pos]
            ).sum()
            u = float(u_min[pos, t])
            sign_u = 1.0 if u > 0.0 else -1.0
            value = gi * (-0.5 * sign_u / (1.0 + abs(u)) ** 2)
            x_grad[pos, lut.a_arr[t, r]] += value
            x_grad[pos, lut.b_arr[t, r]] -= value
            expected_s[tree, j_bin] -= learning_rate * grad[pos]
    return x_grad, expected_s


def test_comparisons_per_bit_one_matches_legacy_exactly(monkeypatch):
    args = make_args()
    random.seed(12345)
    expected_a, expected_b = build_comparison_arrays(
        'random', args.n_t, args.n_c, args.embedding_dim)

    random.seed(12345)
    lut = LUT(args.n_t)
    build_LUT(lut, args.n_c, 3, args)
    np.testing.assert_array_equal(lut.a_arr, expected_a)
    np.testing.assert_array_equal(lut.b_arr, expected_b)

    x = np.array([
        [0.2, -0.3, 1.1, 0.7, -0.9, 0.4, 0.8, -0.2],
        [-0.1, 0.5, 0.3, -0.8, 1.2, 0.6, -0.4, 0.9],
    ], dtype=np.float32)
    expected_j, expected_r, expected_u = legacy_cache_index(
        expected_a, expected_b, lut.bitmasks, x)

    monkeypatch.setattr(main_module, 'HAS_NUMBA', False)
    j = np.zeros_like(expected_j)
    r_min = np.zeros_like(expected_r)
    u_min = np.zeros_like(expected_u)
    cache_index_batch(lut, x, j, r_min, u_min)
    np.testing.assert_array_equal(j, expected_j)
    np.testing.assert_array_equal(r_min, expected_r)
    np.testing.assert_array_equal(u_min, expected_u)

    rng = np.random.default_rng(77)
    lut.S[:] = rng.normal(size=lut.S.shape).astype(np.float32)
    grad = rng.normal(size=(x.shape[0], lut.y_dim)).astype(np.float32)
    learning_rate = np.float32(0.01)
    expected_x_grad, expected_s = legacy_backward(
        lut, j, r_min, u_min, grad, learning_rate)

    actual_x_grad = np.zeros_like(expected_x_grad)
    main_module.learning_rate = learning_rate
    _lut_backward_python(
        lut, j, r_min, u_min, grad, actual_x_grad,
        lut.y_dim, x.shape[0], args.n_t, False)
    np.testing.assert_array_equal(actual_x_grad, expected_x_grad)
    np.testing.assert_array_equal(lut.S, expected_s)


@pytest.mark.parametrize('use_numba', [False, True])
def test_xor_parity_and_global_closest_pair(monkeypatch, use_numba):
    if use_numba and not main_module.HAS_NUMBA:
        pytest.skip('Numba is unavailable')
    monkeypatch.setattr(main_module, 'HAS_NUMBA', use_numba)
    lut = LUT(1)
    lut.a_arr = np.array([[0, 2, 4, 6]], dtype=np.int32)
    lut.b_arr = np.array([[1, 3, 5, 7]], dtype=np.int32)
    lut.bitmasks = np.array([1, 2], dtype=np.int32)
    lut.comparisons_per_bit = 2
    x = np.array([[3.0, 1.0, 0.0, 1.0, 1.1, 1.0, 4.0, 2.0]],
                 dtype=np.float32)
    j = np.zeros((1, 1), dtype=np.int32)
    r_min = np.zeros((1, 1), dtype=np.int32)
    u_min = np.zeros((1, 1), dtype=np.float32)

    cache_index_batch(lut, x, j, r_min, u_min)

    np.testing.assert_array_equal(j, np.array([[1]], dtype=np.int32))
    np.testing.assert_array_equal(r_min, np.array([[2]], dtype=np.int32))
    np.testing.assert_array_equal(u_min, np.array([[x[0, 4] - x[0, 5]]], dtype=np.float32))


def test_backward_flips_aggregate_bit_and_updates_only_closest_pair():
    lut = LUT(1)
    lut.y_dim = 1
    lut.S = np.zeros((1, 4, 1), dtype=np.float32)
    lut.S[0, 3, 0] = 2.0
    lut.a_arr = np.array([[0, 2, 4, 6]], dtype=np.int32)
    lut.b_arr = np.array([[1, 3, 5, 7]], dtype=np.int32)
    lut.bitmasks = np.array([1, 2], dtype=np.int32)
    lut.comparisons_per_bit = 2
    x_grad = np.zeros((1, 8), dtype=np.float32)

    _lut_backward_python(
        lut,
        np.array([[1]], dtype=np.int32),
        np.array([[2]], dtype=np.int32),
        np.array([[0.1]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        x_grad, 1, 1, 1, True)

    expected_value = np.float32(2.0 * (-0.5 / 1.1 ** 2))
    assert x_grad[0, 4] == pytest.approx(expected_value)
    assert x_grad[0, 5] == pytest.approx(-expected_value)
    assert np.count_nonzero(x_grad) == 2


@pytest.mark.parametrize('value', [0, -1])
def test_comparisons_per_bit_must_be_positive(value):
    with pytest.raises(ValueError, match='at least 1'):
        configure_lut_comparison_mode(
            make_args(comparisons_per_bit=value), ParserStub())


@pytest.mark.parametrize('mode', ['overlap', 'no-overlap'])
def test_xor_fan_in_rejects_non_random_modes(mode):
    with pytest.raises(ValueError, match='requires.*random'):
        configure_lut_comparison_mode(
            make_args(comparisons_per_bit=2, lut_comparison_mode=mode),
            ParserStub())


def test_xor_checkpoint_round_trip_and_legacy_shape_compatibility():
    for comparisons_per_bit in (1, 3):
        args = make_args(comparisons_per_bit=comparisons_per_bit)
        source = LUT(args.n_t)
        target = LUT(args.n_t)
        random.seed(900 + comparisons_per_bit)
        build_LUT(source, args.n_c, 2, args)
        random.seed(1000 + comparisons_per_bit)
        build_LUT(target, args.n_c, 2, args)
        source.S[:] = np.arange(source.S.size, dtype=np.float32).reshape(source.S.shape)

        checkpoint = {}
        _lut_save(checkpoint, 'lut', source)
        _lut_load(checkpoint, 'lut', target)

        np.testing.assert_array_equal(target.S, source.S)
        np.testing.assert_array_equal(target.a_arr, source.a_arr)
        np.testing.assert_array_equal(target.b_arr, source.b_arr)
        assert target.a_arr.shape == (
            args.n_t, args.n_c * comparisons_per_bit)
