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
from lut_torch.data import IGNORE_INDEX, UNK_TOKEN_ID, SnippetDataset
from lut_torch.modules import LutModel
from lut_torch.train import (
    best_model_path,
    build_optimizers,
    build_parser,
    checkpoint_path,
    derive_schedule,
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

@parametrize('attention', ['new', 'hard'])
def test_batch_equals_accumulated_singles(attention):
    """A batch of B must give the same gradients as B single-sequence steps averaged.

    This is the property that lets a real batch dimension replace main.py's
    BatchElement + SparseGradAccumulator + apply_averaged_gradients replay.
    """
    model, cfg = make_model(attention=attention)
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
# Epoch budget
# ---------------------------------------------------------------------------

class _Corpus:
    """Stands in for SnippetDataset: derive_schedule only reads `.length`."""

    def __init__(self, length):
        self.length = length


class _Parser:
    def error(self, message):          # argparse's contract: never returns
        raise AssertionError(message)


def _derived(length, extra):
    args = parse_args(extra)
    derive_schedule(args, _Corpus(length), _Parser())
    return args


@parametrize('batch', [1, 8, 32])
def test_epoch_budget_equalises_data_across_batch_sizes(batch):
    """Every batch size must see the same tokens, which --max-steps cannot deliver."""
    length = 2_156_877
    args = _derived(length, ['--batch-size', str(batch), '--epochs', '10',
                             '--context-size', '32'])
    # max_steps is a ceil, so the budget is met or just cleared, never undercut, and
    # never by more than the one step the rounding can add.
    assert 0.0 <= args.epochs_run - 10.0 < 1.0 / args.steps_per_epoch
    seen = args.max_steps * args.tokens_per_step
    assert abs(seen / length - 10.0) < 0.01, seen


def test_warmup_frac_holds_warmup_tokens_and_depth_constant():
    """Without the floor, the fraction is what makes warmup mean the same thing."""
    tokens, depths = [], []
    for batch in (1, 8, 32):
        args = _derived(2_156_877, ['--batch-size', str(batch), '--epochs', '10',
                                    '--context-size', '32', '--warmup-frac', '0.0015',
                                    '--warmup-min-steps', '1'])
        tokens.append(args.warmup_steps * args.tokens_per_step)
        depths.append(args.lr_decay_depth)
    # Warmup lands on a whole number of steps, so the token counts agree only to within
    # one step of the largest batch -- 3% here, against the 32x spread a fixed count
    # would give.
    assert max(tokens) / min(tokens) < 1.05, tokens
    assert max(depths) / min(depths) < 1.05, depths


def test_warmup_min_steps_is_a_floor_only():
    """The floor may raise warmup on a short run; it must never lower it on a long one."""
    long_run = _derived(2_156_877, ['--batch-size', '1', '--epochs', '10',
                                    '--context-size', '32', '--warmup-frac', '0.0015',
                                    '--warmup-min-steps', '200'])
    short_run = _derived(2_156_877, ['--batch-size', '32', '--epochs', '10',
                                     '--context-size', '32', '--warmup-frac', '0.0015',
                                     '--warmup-min-steps', '200'])
    assert long_run.warmup_steps == 1011        # the fraction wins
    assert short_run.warmup_steps == 200        # the floor wins


@parametrize('mode,expected',
             [('none', 0.01), ('linear', 0.32), ('sqrt', 0.01 * 32 ** 0.5)])
def test_lr_batch_scaling(mode, expected):
    args = _derived(2_156_877, ['--batch-size', '32', '--epochs', '10',
                                '--context-size', '32', '--max-learning-rate', '0.01',
                                '--lr-batch-scaling', mode])
    assert abs(args.max_learning_rate - expected) < 1e-9
    assert args.base_learning_rate == 0.01      # the quoted value is preserved


def test_reference_batch_size_leaves_that_batch_alone():
    args = _derived(2_156_877, ['--batch-size', '1', '--epochs', '10',
                                '--context-size', '32', '--max-learning-rate', '0.01',
                                '--lr-batch-scaling', 'linear'])
    assert args.max_learning_rate == 0.01


@parametrize('batch', [1, 8, 32])
def test_epoch_cadence_gives_every_run_the_same_points(batch):
    args = _derived(2_156_877, ['--batch-size', str(batch), '--epochs', '10',
                                '--context-size', '32',
                                '--validation-interval-epochs', '0.1'])
    assert 98 <= args.max_steps // args.validation_interval <= 102


# ---------------------------------------------------------------------------
# LR cooldown
# ---------------------------------------------------------------------------

def test_cooldown_reaches_zero_and_is_monotonic_in_the_tail():
    args = parse_args(['--warmup-steps', '10', '--max-steps', '1000',
                       '--lr-cooldown-frac', '0.2'])
    factor = lr_lambda_factory(args)
    assert factor(args.max_steps) == 0.0
    tail = [factor(s) for s in range(800, 1001, 20)]
    assert all(a > b for a, b in zip(tail, tail[1:])), tail
    # Untouched before the cooldown begins.
    plain = lr_lambda_factory(parse_args(['--warmup-steps', '10', '--max-steps', '1000']))
    assert all(abs(factor(s) - plain(s)) < 1e-12 for s in range(0, 800))


def test_decay_start_defaults_to_the_historical_schedule():
    """t0 == warmup must leave the curve untouched, or every past run is reinterpreted."""
    plain = lr_lambda_factory(parse_args(['--warmup-steps', '50', '--max-steps', '2000']))
    same = lr_lambda_factory(parse_args(['--warmup-steps', '50', '--max-steps', '2000',
                                         '--lr-decay-start', '50']))
    assert all(abs(plain(s) - same(s)) < 1e-12 for s in range(2000))


def test_decay_start_holds_the_peak_then_decays():
    """The point of the knob: a flat stretch the warmup length no longer dictates."""
    args = parse_args(['--warmup-steps', '50', '--max-steps', '2000',
                       '--lr-decay-start', '500'])
    factor = lr_lambda_factory(args)
    assert abs(factor(50) - 1.0) < 1e-12          # ramp ends at the peak
    assert all(abs(factor(s) - 1.0) < 1e-12 for s in range(50, 501))   # ...and holds
    assert factor(1000) < 1.0                                          # ...then decays
    assert factor(2000) < factor(1000)


def test_decay_start_sets_depth_independently_of_the_ramp():
    """Two ramps, one decay curve -- which is what makes a warmup comparison mean
    something. Under the old coupling these two would differ everywhere after warmup."""
    short = lr_lambda_factory(parse_args(['--warmup-steps', '20', '--max-steps', '2000',
                                          '--lr-decay-start', '500']))
    long = lr_lambda_factory(parse_args(['--warmup-steps', '200', '--max-steps', '2000',
                                         '--lr-decay-start', '500']))
    assert short(10) != long(10)                                    # ramps differ
    assert all(abs(short(s) - long(s)) < 1e-12 for s in range(500, 2000))


def test_decay_start_below_warmup_is_refused():
    args = parse_args(['--warmup-steps', '200', '--epochs', '1', '--context-size', '32',
                       '--batch-size', '32', '--lr-decay-start', '50'])
    try:
        derive_schedule(args, _Corpus(2_156_877), _Parser())
    except AssertionError as exc:                  # _Parser.error raises AssertionError
        assert 'lr-decay-start' in str(exc)
    else:
        raise AssertionError('expected derive_schedule to reject t0 < warmup')


def test_cooldown_min_ratio_is_the_floor():
    args = parse_args(['--warmup-steps', '10', '--max-steps', '1000',
                       '--lr-cooldown-frac', '0.5', '--lr-cooldown-min-ratio', '0.1'])
    factor = lr_lambda_factory(args)
    plain = lr_lambda_factory(parse_args(['--warmup-steps', '10', '--max-steps', '1000']))
    assert abs(factor(args.max_steps) - 0.1 * plain(args.max_steps)) < 1e-12


def test_checkpoint_interval_thins_only_the_periodic_save():
    """The 7.1 GB models validate 100 times in a 10-epoch run; the periodic checkpoint
    is the one that can be thinned, because the best copy is written only on
    improvement and the final one is written after the loop either way."""
    args = parse_args(['--checkpoint-interval-validations', '10'])
    assert args.checkpoint_interval_validations == 10
    kept = [v for v in range(1, 101) if v % args.checkpoint_interval_validations == 0]
    assert kept == list(range(10, 101, 10))          # 10 writes instead of 100
    assert parse_args([]).checkpoint_interval_validations == 1   # default unchanged


def test_best_model_path_is_a_sibling():
    assert checkpoint_path(best_model_path('out/model')) == 'out/model_best.pt'
    assert checkpoint_path(best_model_path('out/model.pt')) == 'out/model_best.pt'


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@parametrize('optimizer', ['sgd', 'adam'])
@parametrize('attention', ['new', 'hard'])
def test_checkpoint_round_trip(optimizer, attention):
    """Saving and reloading must restore weights, optimizer, schedule and RNG."""
    model, cfg = make_model(attention=attention)
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

        restored, _ = make_model(seed=99, attention=attention)  # different init on purpose
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
# Tree weights
# ---------------------------------------------------------------------------

@parametrize('stack_heads', [False, True])
def test_tree_weights_start_as_the_unweighted_model(stack_heads):
    """Ones-initialised weights must not change the forward or the table gradients."""
    rng = np.random.default_rng(12)
    inputs = rng.integers(0, CFG_KWARGS['vocab_size_input'], (2, CFG_KWARGS['context_size']))
    targets = rng.integers(0, CFG_KWARGS['vocab_size_output'], (2, CFG_KWARGS['context_size']))
    results = []
    for tree_weights in (False, True):
        model, cfg = make_model(tree_weights=tree_weights, stack_heads=stack_heads,
                                table_grad='dense')
        loss = objective(model, cfg, inputs, targets)
        loss.backward()
        results.append((float(loss), table_grads(model), model))
    (loss_a, grads_a, _), (loss_b, grads_b, weighted) = results
    assert abs(loss_a - loss_b) <= 1e-5 * abs(loss_a)
    for a, b in zip(grads_a, grads_b):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    weights = weighted.tree_weight_parameters()
    assert weights and all(torch.equal(w.detach(), torch.ones_like(w)) for w in weights)
    assert all(w.grad is not None and bool(w.grad.abs().sum() > 0) for w in weights)


def test_every_table_gets_tree_weights():
    model, cfg = make_model(tree_weights=True)
    plain, _ = make_model()
    assert plain.tree_weight_parameters() == []
    L, H = cfg.num_layers, cfg.num_heads
    # Q, K, V, Final per head, plus FFN and MHA_O per layer, plus the output head.
    assert len(model.tree_weight_parameters()) == L * (4 * H + 2) + 1


@parametrize('optimizer', ['sgd', 'adam'])
def test_tree_weights_train_in_their_own_group(optimizer):
    """Same lr as the dense group, no weight decay, and every parameter in exactly one
    group of one optimizer."""
    model, cfg = make_model(tree_weights=True)
    args = parse_args(['--optimizer', optimizer, '--warmup-steps', '2', '--max-steps', '20',
                       '--weight-decay', '0.1', '--max-learning-rate', '0.05',
                       '--tree-weights'])
    optimizers, schedulers = build_optimizers(model, args)
    for sched in schedulers:      # warmup starts at lr 0, which would hide the step
        sched.step()
    seen = [id(p) for opt in optimizers for g in opt.param_groups for p in g['params']]
    assert sorted(seen) == sorted(id(p) for p in model.parameters() if p.requires_grad)

    tree_ids = {id(p) for p in model.tree_weight_parameters()}
    groups = [g for opt in optimizers for g in opt.param_groups
              if {id(p) for p in g['params']} & tree_ids]
    assert len(groups) == 1 and {id(p) for p in groups[0]['params']} == tree_ids
    assert groups[0]['weight_decay'] == 0.0
    assert groups[0]['lr'] == optimizers[-1].param_groups[0]['lr']

    rng = np.random.default_rng(13)
    inputs = rng.integers(0, cfg.vocab_size_input, (2, cfg.context_size))
    targets = rng.integers(0, cfg.vocab_size_output, (2, cfg.context_size))
    objective(model, cfg, inputs, targets).backward()
    for opt in optimizers:
        opt.step()
    assert any(not torch.equal(w.detach(), torch.ones_like(w))
               for w in model.tree_weight_parameters())


def test_tree_weights_survive_a_checkpoint():
    model, cfg = make_model(tree_weights=True)
    with torch.no_grad():
        for w in model.tree_weight_parameters():
            w.uniform_(0.5, 1.5)
    args = parse_args(['--warmup-steps', '2', '--max-steps', '20', '--tree-weights'])
    optimizers, schedulers = build_optimizers(model, args)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'ckpt')
        save_checkpoint(path, model, cfg, optimizers, schedulers, 3, args)
        restored, _ = make_model(seed=99, tree_weights=True)
        r_opts, r_scheds = build_optimizers(restored, args)
        load_checkpoint(path, restored, r_opts, r_scheds, reset_optimizer=False)
    for a, b in zip(model.tree_weight_parameters(), restored.tree_weight_parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@parametrize('pos_task', [True, False])
@parametrize('last_only', [False, True])
def test_evaluate_matches_manual_scoring(pos_task, last_only):
    """evaluate() scores every valid position, or only the last under the LM flag.

    ``--lm-validate-last-position-only`` is LM-only, so the POS cells must be unaffected
    by it -- POS has always scored every position and still does.
    """
    model, cfg = make_model(POS_task=pos_task,
                            vocab_size_output=7 if pos_task else 23)
    model.eval()
    rng = np.random.default_rng(4)
    tokens = rng.integers(1, cfg.vocab_size_input, 200)
    tokens[3::7] = UNK_TOKEN_ID     # so the <unk>-excluded split is actually exercised
    pos = rng.integers(0, 7, 200) if pos_task else None
    dataset = SnippetDataset(tokens, pos, cfg.context_size, pos_task)
    starts = np.arange(6)
    argv = ['--validation-batch-size', '3']
    if last_only:
        argv.append('--lm-validate-last-position-only')
    args = parse_args(argv)

    val = evaluate(model, dataset, starts, args, cfg, torch.device('cpu'))
    loss, acc = val['loss'], val['acc']

    scored_last_only = last_only and not pos_task
    total, correct, count = 0.0, 0, 0
    nu_total, nu_correct, nu_count = 0.0, 0, 0
    with torch.no_grad():
        for start in starts:
            inp, tgt = dataset.batch([start])
            logits = model(torch.from_numpy(inp))[0]
            positions = ([cfg.context_size - 1] if scored_last_only
                         else range(cfg.context_size))
            for p in positions:
                if tgt[0, p] == IGNORE_INDEX:
                    continue
                item = float(F.cross_entropy(logits[p:p + 1],
                                             torch.tensor([int(tgt[0, p])])))
                hit = int(int(logits[p].argmax()) == int(tgt[0, p]))
                total += item
                correct += hit
                count += 1
                if not pos_task and int(tgt[0, p]) != UNK_TOKEN_ID:
                    nu_total += item
                    nu_correct += hit
                    nu_count += 1
    assert count > 0
    assert abs(loss - total / count) < 1e-4
    assert abs(acc - correct / count) < 1e-9

    # The <unk>-excluded pair is LM-only and must be scored over a strict subset.
    if pos_task:
        assert val['count_no_unk'] == 0
        assert val['loss_no_unk'] != val['loss_no_unk']      # NaN
    else:
        assert val['count_no_unk'] == nu_count
        # Only the all-positions mode is guaranteed to score an <unk> target: last-only
        # scores 6 tokens, and whether one of them lands on the 3::7 injection depends
        # on the context size.
        if not scored_last_only:
            assert nu_count < val['count']
        if nu_count:
            assert abs(val['loss_no_unk'] - nu_total / nu_count) < 1e-4
            assert abs(val['acc_no_unk'] - nu_correct / nu_count) < 1e-9


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
