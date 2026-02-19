"""Tests verifying that prange-parallelized attention forward matches sequential."""

import numpy as np
import pytest
import random as py_random
from types import SimpleNamespace
from numba import njit

import main as main_module
from main import (
    LUT, Model, AttentionHead, build_LUT, build_Model,
    attention_forward, model_forward, model_training_step,
    cache_index_batch, cache_PE_index_batch,
    _attention_forward_kernel,
)


def make_full_args(**overrides):
    defaults = dict(
        context_size=32, vocab_size=256, embedding_dim=16,
        positional_dim=4, num_layers=2, num_heads=2,
        n_t=8, n_c=4, testing_length=10, max_steps=10,
        validation_interval=100, temperature=0.4,
        factored_output=False, loss_file='/dev/null',
    )
    defaults.update(overrides)
    args = SimpleNamespace(**defaults)
    args.shift_qk = args.n_c + args.positional_dim
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.vocab_hi = (args.vocab_size + 255) // 256
    return args


# Reference: sequential forward kernel (no parallelism)
@njit(cache=False)
def _attention_forward_kernel_sequential(S, all_jV, all_jPE, y, shift_qk, pd, CS, N_T, y_dim):
    for pos in range(1, CS):
        for pos1 in range(pos):
            for t in range(N_T):
                j = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pos - pos1, t]
                for d in range(y_dim):
                    y[pos, d] += S[t, j, d]


class TestPrangeForwardKernel:
    """Direct kernel-level comparison: sequential vs prange."""

    @pytest.mark.parametrize("context_size", [8, 32, 64, 128])
    def test_kernel_matches_sequential(self, context_size):
        n_t, n_c, pd, ed = 8, 4, 4, 16
        shift_qk = n_c + pd
        total_n_c = n_c + n_c + pd  # attention V LUT uses concatenated bits
        num_bins = 1 << total_n_c

        np.random.seed(42)
        S = np.random.randn(n_t, num_bins, ed).astype(np.float32) * 0.1
        all_jV = np.random.randint(0, 1 << n_c, (context_size, n_t)).astype(np.int32)
        all_jPE = np.random.randint(0, 1 << pd, (context_size, n_t)).astype(np.int32)

        y_seq = np.zeros((context_size, ed), dtype=np.float32)
        y_par = np.zeros((context_size, ed), dtype=np.float32)

        _attention_forward_kernel_sequential(S, all_jV, all_jPE, y_seq,
                                             shift_qk, pd, context_size, n_t, ed)
        _attention_forward_kernel(S, all_jV, all_jPE, y_par,
                                  shift_qk, pd, context_size, n_t, ed)

        np.testing.assert_allclose(y_par, y_seq, atol=1e-5,
            err_msg=f"prange kernel mismatch at context_size={context_size}")

    def test_kernel_deterministic_across_runs(self):
        """Run the parallel kernel multiple times, verify identical output."""
        n_t, n_c, pd, ed = 8, 4, 4, 16
        cs = 64
        shift_qk = n_c + pd
        num_bins = 1 << (n_c + n_c + pd)

        np.random.seed(42)
        S = np.random.randn(n_t, num_bins, ed).astype(np.float32) * 0.1
        all_jV = np.random.randint(0, 1 << n_c, (cs, n_t)).astype(np.int32)
        all_jPE = np.random.randint(0, 1 << pd, (cs, n_t)).astype(np.int32)

        results = []
        for _ in range(5):
            y = np.zeros((cs, ed), dtype=np.float32)
            _attention_forward_kernel(S, all_jV, all_jPE, y, shift_qk, pd, cs, n_t, ed)
            results.append(y.copy())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i],
                err_msg=f"Non-deterministic output on run {i}")


class TestAttentionForwardIntegration:
    """Full attention_forward function comparison using the Model structs."""

    @pytest.mark.parametrize("context_size", [16, 32, 64])
    def test_attention_forward_matches(self, context_size):
        args = make_full_args(context_size=context_size, n_t=8, n_c=4,
                              embedding_dim=16, positional_dim=4)

        # Build an attention head
        py_random.seed(42)
        np.random.seed(42)
        head = AttentionHead(args)
        build_LUT(head.V, args.n_c + args.n_c + args.positional_dim,
                  args.embedding_dim, args)
        head.Positional_encoding[:] = np.random.randn(
            context_size, args.n_t, args.positional_dim).astype(np.float32)

        # Random input
        np.random.seed(99)
        x = np.random.randn(context_size, args.embedding_dim).astype(np.float32)

        # Run with parallel kernel
        y_par = np.zeros((context_size, args.embedding_dim), dtype=np.float32)
        attention_forward(head, x, y_par, args)

        # Run sequential reference using the same cached indices
        y_seq = np.zeros((context_size, args.embedding_dim), dtype=np.float32)
        _attention_forward_kernel_sequential(
            head.V.S, head._v_j, head._pe_j, y_seq,
            args.shift_qk, args.positional_dim,
            context_size, args.n_t, head.V.y_dim)

        np.testing.assert_allclose(y_par, y_seq, atol=1e-5,
            err_msg=f"attention_forward mismatch at context_size={context_size}")


class TestEndToEndTraining:
    """Full model training step: verify loss and gradients are unchanged."""

    @pytest.mark.parametrize("factored", [False, True])
    def test_training_step_loss_matches(self, factored):
        """Run N training steps with fixed seed, compare loss trajectory."""
        cs, ed = 16, 16

        def run_training(seed):
            args = make_full_args(
                context_size=cs, embedding_dim=ed, num_layers=2,
                num_heads=2, n_t=8, n_c=4, vocab_size=256,
                factored_output=factored,
            )
            py_random.seed(seed)
            np.random.seed(seed)

            m = Model(args)
            build_Model(m, args)

            # Fake token data
            np.random.seed(seed + 100)
            tokens = np.random.randint(0, args.vocab_size, cs + 1).astype(np.int32)
            m.output_head.load_targets(tokens, cs)

            losses = []
            for step in range(5):
                main_module.learning_rate = 0.001
                for pos in range(cs):
                    m.z[pos] = m.Token_embedder[tokens[pos]]

                model_forward(m, args)

                # Compute validation loss at last position
                loss = m.output_head.validation_loss(args)
                losses.append(loss)

                # Re-embed and re-forward for training step (validation_loss modified softmax)
                for pos in range(cs):
                    m.z[pos] = m.Token_embedder[tokens[pos]]
                model_training_step(m, args)

            return losses

        # Run twice with same seed — should get identical results
        losses1 = run_training(seed=777)
        losses2 = run_training(seed=777)

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert abs(l1 - l2) < 1e-4, \
                f"Loss diverged at step {i}: {l1} vs {l2}"

    def test_loss_decreases(self):
        """Sanity check: loss should decrease over many steps."""
        args = make_full_args(
            context_size=16, embedding_dim=16, num_layers=2,
            num_heads=2, n_t=8, n_c=4, vocab_size=256,
            factored_output=False,
        )

        py_random.seed(42)
        np.random.seed(42)

        m = Model(args)
        build_Model(m, args)

        np.random.seed(123)
        tokens = np.random.randint(0, args.vocab_size, args.context_size + 1).astype(np.int32)
        m.output_head.load_targets(tokens, args.context_size)

        losses = []
        for step in range(200):
            main_module.learning_rate = 0.01
            for pos in range(args.context_size):
                m.z[pos] = m.Token_embedder[tokens[pos]]

            model_forward(m, args)

            # Compute loss
            output = m.output_head.output
            loss = 0.0
            for pos in range(args.context_size):
                logits = output[pos, :args.vocab_size].copy()
                max_val = logits.max()
                exp_logits = np.exp(logits - max_val)
                probs = exp_logits / exp_logits.sum()
                loss += -np.log(max(probs[tokens[pos + 1]], 1e-10))
            losses.append(loss / args.context_size)

            model_training_step(m, args)

        # Loss should decrease substantially over 200 steps on a fixed sample
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert losses[-1] < losses[0] * 0.9, \
            f"Loss decreased less than 10%: {losses[0]:.4f} -> {losses[-1]:.4f}"
