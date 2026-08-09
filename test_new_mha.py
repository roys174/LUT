from multiprocessing.pool import ThreadPool
from types import SimpleNamespace
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
    BatchElement,
    Model,
    apply_averaged_gradients,
    build_Model,
    configure_new_mha,
    configure_weight_sharing,
    model_backward,
    model_forward,
)


class ParserStub:
    def error(self, message):
        raise ValueError(message)


def make_args(**overrides):
    args = SimpleNamespace(
        vocab_size_input=16,
        vocab_size_output=16,
        context_size=4,
        embedding_dim=8,
        positional_dim=1,
        num_layers=1,
        num_heads=2,
        n_t=2,
        n_c=2,
        weight_sharing=None,
        lut_comparison_mode='random',
        attn_qk_dim=2,
        attn_qk_n_c=1,
        attn_qk_n_t=2,
        factored_output=False,
        fp16=False,
        new_attention=False,
        new_MHA=True,
        soft_attention=False,
        PE_only=False,
        freeze_PE=False,
        PE_init_scale=1.0,
        qkpe_direct_sign=False,
        normalize_qk=False,
        qk_norm_eps=1e-6,
        qk_norm_target_std=np.float32(1.0 / np.sqrt(3.0)),
        layernorm=False,
        dropout=0.0,
        POS_task=False,
        vocab_hi=1,
        vocab_lo=256,
        temperature=1.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    configure_new_mha(args, ParserStub())
    configure_weight_sharing(args, ParserStub())
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim
    return args


def randomize_lut_tables(m, seed=0):
    rng = np.random.default_rng(seed)
    for lut in m.FFN:
        lut.S[:] = rng.normal(0.0, 0.05, size=lut.S.shape).astype(lut.S.dtype)
    if m.MHA_O is not None:
        for lut in m.MHA_O:
            lut.S[:] = rng.normal(0.0, 0.05, size=lut.S.shape).astype(lut.S.dtype)
    for layer in m.head:
        for head in layer:
            for lut in (head.Q, head.K, head.V, head.Final):
                lut.S[:] = rng.normal(0.0, 0.05, size=lut.S.shape).astype(lut.S.dtype)
    m.output_head.unembedder.S[:] = rng.normal(
        0.0, 0.05, size=m.output_head.unembedder.S.shape
    ).astype(m.output_head.unembedder.S.dtype)


def load_tokens(model, args, offset=0):
    tokens = (np.arange(args.context_size + 1, dtype=np.int32) + offset) % args.vocab_size_output
    model.tokens[:] = tokens
    model.z[:] = model.Token_embedder[tokens[:args.context_size]]
    model.output_head.load_targets(tokens, args.context_size)


def test_new_mha_builds_lower_dim_heads_and_o_lut(monkeypatch):
    monkeypatch.setattr(main_module, 'HAS_NUMBA', False)
    args = make_args()
    model = Model(args)
    build_Model(model, args)

    assert args.new_attention is True
    assert args.mha_head_dim == 4
    assert model.MHA_O[0].y_dim == args.embedding_dim
    assert model.MHA_O[0].S.shape[0] == args.n_t
    for head in model.head[0]:
        assert head.V.y_dim == args.embedding_dim
        assert head.Final.y_dim == args.mha_head_dim
        assert head._v_embed.shape == (args.context_size, args.embedding_dim)

    load_tokens(model, args)
    model_forward(model, args, training=False)
    assert model.z.shape == (args.context_size, args.embedding_dim)
    assert model._mha_concat[0].shape == (args.context_size, args.embedding_dim)


def test_new_mha_parallel_head_forward_matches_sequential(monkeypatch):
    monkeypatch.setattr(main_module, 'HAS_NUMBA', False)
    args = make_args()
    model = Model(args)
    build_Model(model, args)
    randomize_lut_tables(model)
    load_tokens(model, args)
    z0 = model.z.copy()

    model_forward(model, args, training=False)
    sequential = model.z.copy()

    model.z[:] = z0
    with ThreadPool(2) as pool:
        model_forward(model, args, training=False, head_pool=pool)

    np.testing.assert_allclose(model.z, sequential, rtol=0.0, atol=0.0)


def test_new_mha_rejects_bad_dimension_and_soft_attention():
    with pytest.raises(ValueError, match='divisible'):
        make_args(embedding_dim=7)
    with pytest.raises(ValueError, match='soft-attention'):
        make_args(soft_attention=True)


def test_new_mha_preserves_final_weight_sharing(monkeypatch):
    monkeypatch.setattr(main_module, 'HAS_NUMBA', False)
    args = make_args(weight_sharing=1)
    model = Model(args)
    build_Model(model, args)

    for head in model.head[0]:
        assert head.Final.S.shape[0] == 1
        np.testing.assert_array_equal(head.Final.trees, np.array([0, 0], dtype=np.int32))
    assert model.MHA_O[0].S.shape[0] == args.n_t
    np.testing.assert_array_equal(model.MHA_O[0].trees, np.arange(args.n_t, dtype=np.int32))


def test_new_mha_backward_and_sparse_replay_smoke(monkeypatch):
    monkeypatch.setattr(main_module, 'HAS_NUMBA', False)
    args = make_args(weight_sharing=1)
    model = Model(args)
    build_Model(model, args)
    randomize_lut_tables(model)
    main_module.learning_rate = 0.01

    load_tokens(model, args)
    model_forward(model, args, training=True)
    model.output_head.compute_gradients(args)
    model_backward(model, args)

    batch_model = Model(args)
    build_Model(batch_model, args)
    randomize_lut_tables(batch_model, seed=1)
    batch_elements = [BatchElement(batch_model, args) for _ in range(2)]
    for i, be in enumerate(batch_elements):
        load_tokens(be, args, offset=i)
        model_forward(be, args, training=True)
        be.output_head.compute_gradients(args)
        model_backward(be, args, grad_accum=be.grad_accum)

    before_o = batch_model.MHA_O[0].S.copy()
    apply_averaged_gradients(batch_model, batch_elements, 0.01, args)

    assert np.any(batch_model.MHA_O[0].S != before_o)
    assert all(be.grad_accum.attn_y_grads[0] is not None for be in batch_elements)
    assert all(be.grad_accum.new_attn_final_y_grads[0][h] is not None
               for be in batch_elements for h in range(args.num_heads))
    assert batch_model.MHA_O[0].S.shape == before_o.shape
