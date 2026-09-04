"""Training entry point for the torch LUT model.

    python -m lut_torch.train TRAIN_FILE [options]

Differences from ``main.py``'s loop, all of them deliberate (see the refactor plan):

* real batching instead of ``BatchElement`` + sparse-gradient replay + thread pools;
* torch optimizers (SGD for parity, Adam/AdamW with ``SparseAdam`` for the tables);
* the LR schedule as a ``LambdaLR``;
* ``state_dict`` checkpoints that actually resume (step, optimizer, scheduler, RNG).

The training objective reproduces ``main.py`` exactly: its ``compute_gradients`` writes
``softmax - onehot`` with no ``1/CS``, and batch replay scales by ``lr / batch_size``, so
the backward objective is the *sum* of per-position cross entropy divided by the batch
size, while the number reported is the mean over valid positions.
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ConfigError, LutConfig
from .data import IGNORE_INDEX, load_datasets
from .functional import (
    SELECTION_CLOSEST,
    SELECTION_MODES,
    TABLE_GRAD_DENSE,
    TABLE_GRAD_MODES,
    TABLE_GRAD_SPARSE,
)
from .modules import LutModel

DEFAULT_WARMUP_STEPS = 4000
DEFAULT_MAX_LEARNING_RATE = 1.0 / math.sqrt(4001)
DEFAULT_INVERSE_SQRT_ALPHA = 0.5
DEFAULT_COSINE_MIN_LR_RATIO = 0.01


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='Train the torch LUT model (new attention / new MHA).')

    # data / task
    p.add_argument('training_data', type=str)
    p.add_argument('--validation-data', type=str, default=None)
    p.add_argument('--tokenizer', type=str, default='word')
    p.add_argument('--vocab-file', type=str, default='word_vocab.txt')
    p.add_argument('--POS-task', '--pos-task', dest='POS_task', action='store_true',
                   help='Predict universal POS labels at every position')
    p.add_argument('--pos-cache-dir', type=str, default='.pos_cache')

    # shape
    p.add_argument('--context-size', type=int, default=32)
    p.add_argument('--embedding-dim', type=int, default=64)
    p.add_argument('--positional-dim', type=int, default=5)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--n-t', type=int, default=64)
    p.add_argument('--n-c', type=int, default=6)
    p.add_argument('--comparisons-per-bit', type=int, default=1,
                   help='Random comparisons XORed into each learned LUT bit')
    p.add_argument('--lut-comparison-mode', choices=['random'], default='random',
                   help="Accepted so main.py commands port over; only 'random' exists "
                        "here (no-overlap/overlap were not carried across)")
    p.add_argument('--attn_qk_dim', type=int, default=8)
    p.add_argument('--attn_qk_n_c', type=int, default=6)
    p.add_argument('--attn_qk_n_t', type=int, default=64)

    # architecture / ablations
    p.add_argument('--new-attention', action='store_true',
                   help='Accepted for command-line compatibility; always on here')
    p.add_argument('--new-MHA', '--new-mha', dest='new_MHA', action='store_true')
    p.add_argument('--weight-sharing', type=int, default=None)
    p.add_argument('--PE-only', dest='PE_only', action='store_true')
    p.add_argument('--freeze-PE', dest='freeze_PE', action='store_true')
    p.add_argument('--PE-init-scale', dest='PE_init_scale', type=float, default=1.0)
    p.add_argument('--qkpe-direct-sign', action='store_true')
    p.add_argument('--normalize-qk', action='store_true')
    p.add_argument('--qk-norm-eps', type=float, default=1e-6)
    p.add_argument('--gradient-pair-selection', choices=list(SELECTION_MODES),
                   default=SELECTION_CLOSEST)
    p.add_argument('--table-grad', choices=list(TABLE_GRAD_MODES), default='auto',
                   help="Table-gradient representation. 'auto' uses dense once a table "
                        "is visited at least as many times as it has rows, which is the "
                        "usual case; force 'sparse' only if a dense gradient will not fit")
    p.add_argument('--layernorm', action='store_true')
    p.add_argument('--dropout', type=float, default=0.0)

    # optimization
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=100000)
    p.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw'], default='sgd',
                   help='sgd reproduces main.py; adam/adamw use SparseAdam for tables')
    p.add_argument('--momentum', type=float, default=0.0,
                   help='SGD momentum for dense parameters only (tables stay sparse)')
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='Applied to dense parameters only')
    p.add_argument('--warmup-steps', type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument('--max-learning-rate', type=float,
                   default=DEFAULT_MAX_LEARNING_RATE)
    p.add_argument('--lr-scheduler', choices=['inverse-sqrt', 'cosine'],
                   default='inverse-sqrt')
    p.add_argument('--inverse-sqrt-alpha', type=float,
                   default=DEFAULT_INVERSE_SQRT_ALPHA)
    p.add_argument('--cosine-min-lr-ratio', type=float,
                   default=DEFAULT_COSINE_MIN_LR_RATIO)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--cpu-cores', type=int, default=1,
                   help='torch intra-op thread count')

    # validation / logging / checkpoints
    p.add_argument('--testing-length', type=int, default=10000,
                   help='Validation snippets per evaluation')
    p.add_argument('--validation-interval', type=int, default=10000)
    p.add_argument('--validation-batch-size', type=int, default=0,
                   help='0 uses --batch-size')
    p.add_argument('--loss-file', type=str, default='loss.csv')
    p.add_argument('--save-model', type=str, default=None,
                   help='Checkpoint path; .pt is appended if missing')
    p.add_argument('--load-model', type=str, default=None,
                   help='Resume from a checkpoint (model + optimizer + step + RNG)')
    p.add_argument('--reset-optimizer', action='store_true',
                   help='Load weights only, restarting step/schedule/RNG')

    # W&B
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='lut')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-group', type=str, default=None)
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-tags', nargs='*', default=None)
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                   default='online')
    p.add_argument('--wandb-log-interval', type=int, default=100)
    return p


def validate_args(args, parser):
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    if args.warmup_steps < 1:
        parser.error('--warmup-steps must be at least 1')
    if args.max_learning_rate <= 0.0:
        parser.error('--max-learning-rate must be positive')
    if args.inverse_sqrt_alpha < 0.0:
        parser.error('--inverse-sqrt-alpha must be non-negative')
    if not 0.0 <= args.cosine_min_lr_ratio <= 1.0:
        parser.error('--cosine-min-lr-ratio must be in [0, 1]')
    if args.wandb_log_interval < 1:
        parser.error('--wandb-log-interval must be at least 1')
    if args.validation_interval < 1:
        parser.error('--validation-interval must be at least 1')
    if args.momentum and args.optimizer != 'sgd':
        parser.error('--momentum applies to --optimizer sgd')
    if args.optimizer in ('adam', 'adamw') and args.table_grad == 'auto':
        print("note: --optimizer adam/adamw requires dense table gradients; "
              "--table-grad dense will be used")


# ---------------------------------------------------------------------------
# Schedule and optimizers
# ---------------------------------------------------------------------------

def lr_lambda_factory(args):
    """Multiplier form of ``main.get_learning_rate`` (base LR is the peak LR)."""
    def factor(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        if args.lr_scheduler == 'inverse-sqrt':
            return (args.warmup_steps / step) ** args.inverse_sqrt_alpha
        decay_steps = max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, (step - args.warmup_steps) / decay_steps)
        return (args.cosine_min_lr_ratio + 0.5 * (1.0 - args.cosine_min_lr_ratio)
                * (1.0 + math.cos(math.pi * progress)))
    return factor


def build_optimizers(model, args):
    """Return ``(optimizers, schedulers)``.

    Plain SGD accepts either table-gradient representation, so ``--table-grad auto`` can
    vary per table under it.  Adam/AdamW cannot consume a sparse gradient at all, so
    selecting them pins the tables to dense here -- enforced at this single point, since
    this is where the representation and the optimizer actually have to agree.
    """
    if args.optimizer in ('adam', 'adamw') and args.table_grad != TABLE_GRAD_SPARSE:
        args.table_grad = TABLE_GRAD_DENSE
        model.cfg.table_grad = TABLE_GRAD_DENSE
    tables = [p for p in model.table_parameters() if p.requires_grad]
    dense = model.dense_parameters()
    lr = args.max_learning_rate
    optimizers = []

    if args.optimizer == 'sgd':
        # Plain SGD accepts either representation, so 'auto' can vary per table.
        optimizers.append(torch.optim.SGD([{'params': tables}], lr=lr))
        if dense:
            optimizers.append(torch.optim.SGD(
                [{'params': dense}], lr=lr, momentum=args.momentum,
                weight_decay=args.weight_decay))
    else:
        cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
        if args.table_grad == TABLE_GRAD_SPARSE:
            optimizers.append(torch.optim.SparseAdam(tables, lr=lr))
        else:
            # Adam cannot consume a sparse gradient, so pin the tables to dense.
            optimizers.append(cls(tables, lr=lr, weight_decay=args.weight_decay))
        if dense:
            optimizers.append(cls(dense, lr=lr, weight_decay=args.weight_decay))

    factor = lr_lambda_factory(args)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, factor) for opt in optimizers]
    return optimizers, schedulers


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_path(path):
    return path if path.endswith('.pt') else path + '.pt'


def save_checkpoint(path, model, cfg, optimizers, schedulers, step, args):
    torch.save({
        'step': step,
        'config': cfg.to_dict(),
        'args': {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, bool, str, type(None), list))},
        'model': model.state_dict(),
        'optimizers': [o.state_dict() for o in optimizers],
        'schedulers': [s.state_dict() for s in schedulers],
        'rng': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        },
    }, checkpoint_path(path))


def load_checkpoint(path, model, optimizers, schedulers, reset_optimizer):
    ckpt = torch.load(checkpoint_path(path), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if reset_optimizer:
        print(f"Loaded weights from {checkpoint_path(path)} (optimizer state reset)")
        return 0
    for opt, state in zip(optimizers, ckpt['optimizers']):
        opt.load_state_dict(state)
    for sched, state in zip(schedulers, ckpt['schedulers']):
        sched.load_state_dict(state)
    rng = ckpt.get('rng')
    if rng:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
    step = int(ckpt['step'])
    print(f"Resumed from {checkpoint_path(path)} at step {step}")
    return step


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def batch_loss(model, inputs, targets, vocab_size):
    """Return ``(objective, mean_loss, correct, count)`` for one batch."""
    logits = model(inputs)
    flat = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    total = F.cross_entropy(flat, flat_targets, ignore_index=IGNORE_INDEX,
                            reduction='sum')
    valid = flat_targets != IGNORE_INDEX
    count = int(valid.sum())
    with torch.no_grad():
        correct = int((flat.argmax(dim=-1) == flat_targets)[valid].sum()) if count else 0
    mean = float(total.detach()) / count if count else 0.0
    return total, mean, correct, count


@torch.no_grad()
def evaluate(model, dataset, starts, args, cfg):
    """Validation loss / accuracy.

    POS scores every valid position; language modelling scores only the final position,
    matching ``StandardOutputHead.validation_loss``.
    """
    model.eval()
    batch = args.validation_batch_size or args.batch_size
    loss_sum = 0.0
    correct = 0
    count = 0
    for begin in tqdm(range(0, len(starts), batch), desc='  Val', leave=False,
                      unit='batch'):
        chunk = starts[begin:begin + batch]
        inputs, targets = dataset.batch(chunk)
        if not cfg.POS_task:
            keep = np.full_like(targets, IGNORE_INDEX)
            keep[:, -1] = targets[:, -1]
            targets = keep
        logits = model(torch.from_numpy(inputs))
        flat = logits.reshape(-1, cfg.vocab_size_output)
        flat_targets = torch.from_numpy(targets).reshape(-1)
        valid = flat_targets != IGNORE_INDEX
        if not bool(valid.any()):
            continue
        loss_sum += float(F.cross_entropy(flat[valid], flat_targets[valid],
                                          reduction='sum'))
        correct += int((flat.argmax(dim=-1) == flat_targets)[valid].sum())
        count += int(valid.sum())
    model.train()
    if count == 0:
        return float('nan'), float('nan')
    return loss_sum / count, correct / count


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_cores))

    # Tokenizer and vocabulary sizes.
    if args.tokenizer == 'word':
        from word_tokenizer import POS_LABELS, WordTokenizer
        enc = WordTokenizer(args.vocab_file)
    else:
        import tiktoken
        from word_tokenizer import POS_LABELS
        enc = tiktoken.get_encoding(args.tokenizer)
    args.vocab_size_input = enc.n_vocab
    args.vocab_size_output = len(POS_LABELS) if args.POS_task else enc.n_vocab
    if args.POS_task and args.tokenizer != 'word':
        parser.error('--POS-task requires --tokenizer word')

    try:
        cfg = LutConfig.from_namespace(args)
    except ConfigError as exc:
        parser.error(str(exc))

    print('=' * 50)
    print('Configuration:')
    for key, value in sorted(cfg.to_dict().items()):
        print(f"  {key:24s} = {value}")
    for key in ('batch_size', 'optimizer', 'table_grad', 'momentum', 'weight_decay',
                'max_steps',
                'warmup_steps', 'max_learning_rate', 'lr_scheduler',
                'inverse_sqrt_alpha', 'cosine_min_lr_ratio', 'seed', 'cpu_cores',
                'testing_length', 'validation_interval', 'loss_file', 'save_model',
                'load_model'):
        print(f"  {key:24s} = {getattr(args, key)}")
    print('=' * 50)

    train_set, val_set = load_datasets(args, enc)
    val_starts = np.random.randint(0, val_set.length, size=args.testing_length)

    model = LutModel(cfg)
    print(f"Trainable tables: {len(model.table_parameters())}, "
          f"table entries: {sum(p.numel() for p in model.table_parameters()):,}, "
          f"dense parameters: {sum(p.numel() for p in model.dense_parameters()):,}")

    optimizers, schedulers = build_optimizers(model, args)
    start_step = 0
    if args.load_model:
        start_step = load_checkpoint(args.load_model, model, optimizers, schedulers,
                                     args.reset_optimizer)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            print('Weights & Biases support requires: pip install wandb')
            sys.exit(1)
        config = {k: v for k, v in vars(args).items()
                  if isinstance(v, (int, float, bool, str, type(None), list, tuple))}
        config.update(cfg.to_dict())
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               group=args.wandb_group, entity=args.wandb_entity,
                               tags=args.wandb_tags, mode=args.wandb_mode,
                               config=config)
        wandb.define_metric('step')
        wandb.define_metric('train/*', step_metric='step')
        wandb.define_metric('val/*', step_metric='step')

    with open(args.loss_file, 'w') as f:
        f.write('step, loss, perplexity, accuracy\n')

    model.train()
    ema_loss = None
    ema_alpha = 0.02
    last = {}

    pbar = tqdm(range(start_step, args.max_steps), desc='Training', unit='step')
    for step in pbar:
        starts = np.random.randint(0, train_set.length, size=args.batch_size)
        inputs, targets = train_set.batch(starts)
        objective, mean_loss, _, _ = batch_loss(
            model, torch.from_numpy(inputs), torch.from_numpy(targets),
            cfg.vocab_size_output)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        (objective / args.batch_size).backward()
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

        ema_loss = mean_loss if ema_loss is None else \
            ema_alpha * mean_loss + (1 - ema_alpha) * ema_loss
        lr = optimizers[0].param_groups[0]['lr']

        if wandb_run is not None and step % args.wandb_log_interval == 0:
            wandb_run.log({
                'step': step,
                'train/loss': mean_loss,
                'train/ema_loss': ema_loss,
                'train/perplexity': math.exp(min(ema_loss, 20.0)),
                'train/learning_rate': lr,
            })

        if step % args.validation_interval == 0:
            pbar.set_description('Validating')
            val_loss, val_acc = evaluate(model, val_set, val_starts, args, cfg)
            perplexity = math.exp(min(val_loss, 20.0)) if val_loss == val_loss \
                else float('nan')
            last = {'loss': val_loss, 'ppl': perplexity, 'acc': val_acc}

            with open(args.loss_file, 'a') as f:
                f.write(f"{step}, {val_loss:.6f}, {perplexity:.2f}, {val_acc:.6f}\n")
            tqdm.write(f"\n--- step {step:,} | ppl={perplexity:.2f} | "
                       f"loss={val_loss:.3f} | acc={val_acc:.3f} ---")

            if args.save_model:
                save_checkpoint(args.save_model, model, cfg, optimizers, schedulers,
                                step, args)
                tqdm.write(f"Saved checkpoint to {checkpoint_path(args.save_model)}")
            if wandb_run is not None:
                wandb_run.log({'step': step, 'val/loss': val_loss,
                               'val/perplexity': perplexity, 'val/accuracy': val_acc})
            pbar.set_description('Training')

        postfix = {'lr': f"{lr:.4f}",
                   'train_ppl': f"{math.exp(min(ema_loss, 20.0)):.1f}"}
        if last:
            postfix.update(val_ppl=f"{last['ppl']:.2f}", loss=f"{last['loss']:.3f}",
                           acc=f"{last['acc']:.3f}")
        pbar.set_postfix(**postfix)

    if args.save_model:
        save_checkpoint(args.save_model, model, cfg, optimizers, schedulers,
                        args.max_steps, args)
        print(f"Saved final checkpoint to {checkpoint_path(args.save_model)}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
