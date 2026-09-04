"""Trainer-level tests: batching, LR schedule, checkpoint resume, evaluation.

The parity gate (test_torch_parity.py) proves the per-step maths matches main.py.  These
tests cover the infrastructure that replaced main.py's batch-replay machinery.

Run with:  .venv/bin/python test_torch_training.py
       or: python -m pytest -q test_torch_training.py
"""

import math
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

import main
from lut_torch._testing import parametrize, run_all
from lut_torch.config import LutConfig
from lut_torch.data import IGNORE_INDEX, SnippetDataset
from lut_torch.modules import LutModel
from lut_torch.train import (
    build_optimizers,
    build_parser,
    checkpoint_path,
    evaluate,
    load_checkpoint,
    lr_lambda_factory,
    save_checkpoint,
)

CFG_KWARGS = dict(
    context_size=8, embedding_dim=16, positional_dim=3, num_layers=2, num_heads=2,
    vocab_size_input=23, vocab_size_output=7, n_t=4, n_c=3, attn_qk_dim=4,
    attn_qk_n_c=3, attn_qk_n_t=3, POS_task=True, new_MHA=True, layernorm=True,
)


def make_model(seed=5, **overrides):
    kwargs = dict(CFG_KWARGS)
    kwargs.update(overrides)
    cfg = LutConfig(**kwargs)
    random.seed(seed)
    np.random.seed(seed)
    model = LutModel(cfg)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed)
        for table in model.table_parameters():
            table.copy_(torch.randn(table.shape, generator=gen) * 0.3)
    return model, cfg


def parse_args(extra=()):
    return build_parser().parse_args(['dummy.txt', *extra])


def objective(model, cfg, inputs, targets):
    logits = model(torch.from_numpy(inputs))
    return F.cross_entropy(logits.reshape(-1, cfg.vocab_size_output),
                           torch.from_numpy(targets).reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction='sum')


def table_grads(model):
    out = []
    for table in model.table_parameters():
        grad = table.grad
        if grad is None:
            out.append(np.zeros(tuple(table.shape), dtype=np.float32))
        else:
            out.append((grad.to_dense() if grad.is_sparse else grad).numpy().copy())
    return out


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_batch_equals_accumulated_singles():
    """A batch of B must give the same gradients as B single-sequence steps averaged.

    This is the property that lets a real batch dimension replace main.py's
    BatchElement + SparseGradAccumulator + apply_averaged_gradients replay.
    """
    model, cfg = make_model()
    batch = 4
    rng = np.random.default_rng(2)
    inputs = rng.integers(0, cfg.vocab_size_input, (batch, cfg.context_size))
    targets = rng.integers(0, cfg.vocab_size_output, (batch, cfg.context_size))
    targets[1, 3] = IGNORE_INDEX

    model.zero_grad(set_to_none=True)
    (objective(model, cfg, inputs, targets) / batch).backward()
    batched = table_grads(model)

    model.zero_grad(set_to_none=True)
    for i in range(batch):
        (objective(model, cfg, inputs[i:i + 1], targets[i:i + 1]) / batch).backward()
    singles = table_grads(model)

    for got, want in zip(singles, batched):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


def test_dataset_batch_shapes_and_masking():
    tokens = np.arange(50) % 11
    tokens[7] = 0                                  # UNK_TOKEN_ID
    pos = np.arange(50) % 5
    ds = SnippetDataset(tokens, pos, context_size=6, pos_task=True)
    inputs, targets = ds.batch([0, 3])
    assert inputs.shape == (2, 6) and targets.shape == (2, 6)
    np.testing.assert_array_equal(inputs[0], tokens[:6])
    # Token 7 is <unk>; the window starting at 3 sees it at offset 4, so that POS
    # target must be masked while every other position keeps its label.
    assert targets[1][4] == IGNORE_INDEX
    assert (targets[1][np.arange(6) != 4] != IGNORE_INDEX).all()

    lm = SnippetDataset(tokens, None, context_size=6, pos_task=False)
    lm_inputs, lm_targets = lm.batch([2])
    np.testing.assert_array_equal(lm_targets[0], tokens[3:9])


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

@parametrize('scheduler', ['inverse-sqrt', 'cosine'])
@parametrize('alpha', [0.5, 0.32])
def test_schedule_matches_main(scheduler, alpha):
    args = parse_args(['--lr-scheduler', scheduler, '--warmup-steps', '5',
                       '--max-steps', '40', '--inverse-sqrt-alpha', str(alpha),
                       '--max-learning-rate', '0.01'])
    args.halve_lr_at_20k_40k = False           # main.py flag this port drops

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=args.max_learning_rate)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args))

    for step in range(args.max_steps):
        expected = main.get_learning_rate(step, args)
        assert abs(opt.param_groups[0]['lr'] - expected) < 1e-12, \
            f'step {step}: {opt.param_groups[0]["lr"]} != {expected}'
        opt.step()          # no-op (no gradients), but keeps the documented order
        sched.step()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@parametrize('optimizer', ['sgd', 'adam'])
def test_checkpoint_round_trip(optimizer):
    """Saving and reloading must restore weights, optimizer, schedule and RNG."""
    model, cfg = make_model()
    args = parse_args(['--optimizer', optimizer, '--warmup-steps', '2',
                       '--max-steps', '20', '--max-learning-rate', '0.05'])
    optimizers, schedulers = build_optimizers(model, args)

    rng = np.random.default_rng(9)
    inputs = rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size))
    targets = rng.integers(0, cfg.vocab_size_output, (2, cfg.context_size))

    for _ in range(3):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        (objective(model, cfg, inputs, targets) / 2).backward()
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'ckpt')
        save_checkpoint(path, model, cfg, optimizers, schedulers, 3, args)
        assert os.path.exists(checkpoint_path(path))

        restored, _ = make_model(seed=99)          # different init on purpose
        r_opts, r_scheds = build_optimizers(restored, args)
        step = load_checkpoint(path, restored, r_opts, r_scheds, reset_optimizer=False)
        assert step == 3

        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            torch.testing.assert_close(a, b)
        assert r_opts[0].param_groups[0]['lr'] == optimizers[0].param_groups[0]['lr']

        # Both models must now take an identical next step.
        for m, opts, scheds in ((model, optimizers, schedulers),
                                (restored, r_opts, r_scheds)):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            (objective(m, cfg, inputs, targets) / 2).backward()
            for opt in opts:
                opt.step()
            for sched in scheds:
                sched.step()
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            torch.testing.assert_close(a, b)


def test_checkpoint_reset_optimizer_loads_weights_only():
    model, cfg = make_model()
    args = parse_args(['--warmup-steps', '2', '--max-steps', '20'])
    optimizers, schedulers = build_optimizers(model, args)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'ckpt')
        save_checkpoint(path, model, cfg, optimizers, schedulers, 7, args)
        restored, _ = make_model(seed=99)
        r_opts, r_scheds = build_optimizers(restored, args)
        step = load_checkpoint(path, restored, r_opts, r_scheds, reset_optimizer=True)
        assert step == 0
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@parametrize('pos_task', [True, False])
def test_evaluate_matches_manual_scoring(pos_task):
    """evaluate() must score every valid position for POS, last position only for LM."""
    model, cfg = make_model(POS_task=pos_task,
                            vocab_size_output=7 if pos_task else 23)
    model.eval()
    rng = np.random.default_rng(4)
    tokens = rng.integers(1, cfg.vocab_size_input, 200)
    pos = rng.integers(0, 7, 200) if pos_task else None
    dataset = SnippetDataset(tokens, pos, cfg.context_size, pos_task)
    starts = np.arange(6)
    args = parse_args(['--validation-batch-size', '3'])

    loss, acc = evaluate(model, dataset, starts, args, cfg)

    total, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for start in starts:
            inp, tgt = dataset.batch([start])
            logits = model(torch.from_numpy(inp))[0]
            positions = range(cfg.context_size) if pos_task else [cfg.context_size - 1]
            for p in positions:
                if tgt[0, p] == IGNORE_INDEX:
                    continue
                total += float(F.cross_entropy(logits[p:p + 1],
                                               torch.tensor([int(tgt[0, p])])))
                correct += int(int(logits[p].argmax()) == int(tgt[0, p]))
                count += 1
    assert count > 0
    assert abs(loss - total / count) < 1e-4
    assert abs(acc - correct / count) < 1e-9


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
