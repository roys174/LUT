"""Comparison-pair construction and positional-encoding initialisation.

Every function here is a deliberate, line-for-line reimplementation of its ``main.py``
counterpart, **including the order in which it draws from the RNGs**.  That is what lets
``test_torch_parity.py`` build a ``main.Model`` and a ``lut_torch`` model from the same
seed and get bit-identical ``a_arr``/``b_arr``/PE without any checkpoint interchange.

Draw order matters as much as the formulas: ``random.randint`` (Python RNG) drives the
comparison indices, ``np.random.rand`` (NumPy RNG) drives the embedder and PE.  Do not
"optimise" these loops into vectorised RNG calls -- that would change the stream and break
parity.  ``test_torch_comparisons`` in ``test_torch_lut_primitive.py`` guards this.

Only ``--lut-comparison-mode random`` is supported here; the deterministic ``no-overlap``
and ``overlap`` layouts stay in ``main.py``.
"""

import random

import numpy as np


def fill_vector_with_random_integers(N, max_value):
    """Mirror of ``main.fill_vector_with_random_integers``."""
    return np.array([random.randint(0, max_value - 1) for _ in range(N)], dtype=np.int32)


def fill_vector_with_random_integers_different_from_vector2(vector2, N, max_value):
    """Mirror of ``main.fill_vector_with_random_integers_different_from_vector2``."""
    vector = np.zeros(N, dtype=np.int32)
    for i in range(N):
        while True:
            vector[i] = random.randint(0, max_value - 1)
            if vector[i] != vector2[i]:
                break
    return vector


def build_random_comparison_arrays(n_t, raw_n_c, input_dim):
    """Sample ``n_t`` rows of ``raw_n_c`` distinct-coordinate comparison pairs.

    ``raw_n_c`` is the *raw* comparison count, i.e. ``n_c * comparisons_per_bit`` for
    learned LUT bits (bit-major: raw comparison ``r`` belongs to logical bit ``r // K``).
    The QKPE side passes ``positional_dim`` directly and is never multiplied by ``K``,
    matching ``main.build_Model``.
    """
    a_arr = np.zeros((n_t, raw_n_c), dtype=np.int32)
    b_arr = np.zeros((n_t, raw_n_c), dtype=np.int32)
    for i in range(n_t):
        a_arr[i] = fill_vector_with_random_integers(raw_n_c, input_dim)
        b_arr[i] = fill_vector_with_random_integers_different_from_vector2(
            a_arr[i], raw_n_c, input_dim)
    return a_arr, b_arr


def random_vector(size, scale):
    """Mirror of ``main.random_vector``: uniform in ``[-scale, scale)``."""
    return scale * 2.0 * (np.random.rand(size).astype(np.float32) - 0.5)


def initialize_positional_encoding(shape, pe_init_scale, positional_dim):
    """Mirror of ``main.initialize_positional_encoding``.

    Random magnitudes in ``[0, scale)`` with the sign of dimension ``c`` following the
    binary expansion of the row index (most significant bit first over the first
    ``positional_dim`` dimensions).  Used for ``--freeze-PE``.
    """
    scale = np.float32(pe_init_scale)
    pe = (scale * np.random.rand(*shape)).astype(np.float32)

    signed_dims = min(int(positional_dim), shape[-1])
    if signed_dims > 0:
        row_bins = (np.arange(shape[0], dtype=np.int64) % (1 << signed_dims))
        bit_positions = (signed_dims - 1 - np.arange(signed_dims, dtype=np.int64))
        signs = np.where(
            ((row_bins[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1) != 0,
            np.float32(1.0),
            np.float32(-1.0),
        ).astype(np.float32)
        sign_shape = (shape[0],) + (1,) * (len(shape) - 2) + (signed_dims,)
        pe[..., :signed_dims] *= signs.reshape(sign_shape)
    return pe


def final_tree_map(final_n_t, final_share_m):
    """``--weight-sharing`` map from logical comparator slots to physical Final trees."""
    return np.repeat(np.arange(final_n_t, dtype=np.int32), final_share_m)


def causal_pair_indices(context_size):
    """Flat ``(tok_i, tok_j)`` index vectors for the causal pairs ``j < i``.

    Ordered so that ``pair = i * (i - 1) // 2 + j``, matching ``main.py``'s pair layout.
    Position 0 contributes no pairs, hence receives no attention output.
    """
    tok_i = []
    tok_j = []
    for i in range(1, context_size):
        for j in range(i):
            tok_i.append(i)
            tok_j.append(j)
    return (np.array(tok_i, dtype=np.int64), np.array(tok_j, dtype=np.int64))
