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

# Numba's 'default' resolution picks tbb on this build (bundled tbbpool.so, verified with
# numba.threading_layer()) and falls back to workqueue gracefully where tbb isn't built,
# rather than hard-failing the way pinning NUMBA_THREADING_LAYER=tbb would on a machine
# without it. Either way it does not clash with torch's own intra-op pool. Must be set
# before numba (imported lazily by lut_torch.kernels) JIT-compiles anything.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'default')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import OUTPUT_CODE_MODES, ConfigError, LutConfig
from .data import IGNORE_INDEX, UNK_TOKEN_ID, load_datasets
from .graphs import GraphedTrainStep
from .kernels import KERNEL_BACKEND_MODES, set_kernel_backend, set_kernel_threads
from .functional import (
    SELECTION_CLOSEST,
    SELECTION_MODES,
    TABLE_GRAD_DENSE,
    TABLE_GRAD_MODES,
    TABLE_GRAD_SPARSE,
)
from .modules import LutModel
from .stacking import looks_unstacked, stack_state_dict

DEFAULT_WARMUP_STEPS = 4000
DEFAULT_MAX_LEARNING_RATE = 1.0 / math.sqrt(4001)
DEFAULT_INVERSE_SQRT_ALPHA = 0.5
DEFAULT_COSINE_MIN_LR_RATIO = 0.01
LR_BATCH_SCALING_MODES = ('none', 'linear', 'sqrt')


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
    p.add_argument('--kernels', choices=list(KERNEL_BACKEND_MODES), default='auto',
                   help="LUT execution backend for 'closest' selection on CPU float32. "
                        "'auto' uses fused Numba kernels when importable (several times "
                        "faster than the torch path at batch 1; see lut_torch/AGENTS.md, "
                        "'Kernels'), falling back to torch otherwise. Numerically "
                        "identical either way -- the parity suite runs both.")
    p.add_argument('--stack-heads', dest='stack_heads_mode',
                   choices=['auto', 'on', 'off'], default='auto',
                   help="Run each layer's heads as 4 stacked LUT calls instead of 4*H "
                        "separate ones (5.7x measured on the Final call at the production "
                        "shape; see lut_torch/modules.py, StackedHeads). 'auto' enables it "
                        "whenever there is more than one head. Bit-identical to the "
                        "unstacked path, but it changes the checkpoint layout -- an "
                        "unstacked checkpoint is converted on load, not the other way "
                        "round.")
    p.add_argument('--layernorm', action='store_true')
    p.add_argument('--dropout', type=float, default=0.0)

    # output head
    p.add_argument('--output-code-dim', type=int, default=None,
                   help="Emit D numbers from the output table and map them to the "
                        "vocabulary with a (V, D) code matrix, instead of one score per "
                        "class. Cuts the output table from n_t * 2**n_c * V entries to "
                        "n_t * 2**n_c * D. Default: off (y_dim = V).")
    p.add_argument('--output-code', choices=list(OUTPUT_CODE_MODES), default='learned',
                   help="Which code matrix --output-code-dim uses. 'learned' trains it "
                        "as a dense parameter; 'binary' freezes it to the +-1 binary "
                        "expansion of the class index, which is only meaningful at "
                        "exactly D = ceil(log2(V)) (higher bits of a class index are "
                        "constant, so the code loses rank); 'random' freezes it to an "
                        "i.i.d. +-1 code, which is the one to widen -- expected Hamming "
                        "distance D/2 instead of a minimum of 1.")
    p.add_argument('--tree-weights', action='store_true',
                   help="Give every table a learned weight per tree, so its output is "
                        "sum_t w[t] * S[t, j_t] instead of the plain sum. Initialised to "
                        "1 (a fresh model is unchanged) and trained at the dense learning "
                        "rate without weight decay, which would pull them towards 0 "
                        "rather than 1. Disables the Numba CPU kernels.")
    p.add_argument('--output-code-bias', action='store_true',
                   help="Add a per-class bias to the coded head: logits = y @ C.T + b. "
                        "Without it every logit vector lies in col(C), class priors "
                        "included, which a frozen code cannot choose to accommodate. "
                        "Requires --output-code-dim.")
    p.add_argument('--output-code-lr-scale', type=float, default=1.0,
                   help="Multiply the learning rate of the coded head's C and b by this. "
                        "They take one nominal step where the tables effectively take "
                        "n_t (all n_t visited rows receive the full dL/dy), so at a "
                        "shared LR they train far more slowly. Default 1.0 = the shared "
                        "LR the first runs used.")

    # optimization
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=100000)
    p.add_argument('--epochs', type=float, default=None,
                   help='Budget the run in passes over the training tokens rather than '
                        'in steps: max_steps = ceil(epochs * positions / (batch * CS)). '
                        'A step count is not comparable across batch sizes -- 200k steps '
                        'is 3 passes over this corpus at batch 1 and 96 at batch 32 -- '
                        'so an epoch budget is what puts a batch-size sweep on one axis. '
                        'Overrides --max-steps.')
    p.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw'], default='sgd',
                   help='sgd reproduces main.py; adam/adamw use SparseAdam for tables')
    p.add_argument('--momentum', type=float, default=0.0,
                   help='SGD momentum for dense parameters only (tables stay sparse)')
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='Applied to dense parameters only')
    p.add_argument('--warmup-steps', type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument('--warmup-frac', type=float, default=None,
                   help='Warmup as a fraction of max_steps rather than a fixed count, '
                        'floored at --warmup-min-steps. Under an --epochs budget the '
                        'fraction is what holds warmup to the same number of *tokens* at '
                        'every batch size, and holds the schedule to the same end depth '
                        '(frac ** alpha). Overrides --warmup-steps.')
    p.add_argument('--warmup-min-steps', type=int, default=200,
                   help='Floor for --warmup-frac. A short run makes the fair fraction '
                        'very small (32 steps at batch 32 on a 10-epoch budget), which '
                        'is a coarse ramp into a scaled-up LR; the floor buys a smooth '
                        'ramp at the cost of exact depth matching.')
    p.add_argument('--max-learning-rate', type=float,
                   default=DEFAULT_MAX_LEARNING_RATE)
    p.add_argument('--lr-scheduler', choices=['inverse-sqrt', 'cosine'],
                   default='inverse-sqrt')
    p.add_argument('--inverse-sqrt-alpha', type=float,
                   default=DEFAULT_INVERSE_SQRT_ALPHA)
    p.add_argument('--cosine-min-lr-ratio', type=float,
                   default=DEFAULT_COSINE_MIN_LR_RATIO)
    p.add_argument('--lr-decay-start', type=int, default=None,
                   help='Step at which inverse-sqrt decay begins; the LR sits flat at '
                        'the peak between --warmup-steps and here. Defaults to the '
                        'warmup length, which is the schedule this project has always '
                        'run. Decoupling the two matters because in (W/s)**alpha the '
                        'warmup length also sets the height of the whole decay that '
                        'follows: lengthening the ramp raises the LR at every later '
                        'step, so no warmup comparison is interpretable while they are '
                        'the same number. Must be >= the resolved warmup. inverse-sqrt '
                        'only; ignored by cosine.')
    p.add_argument('--lr-cooldown-frac', type=float, default=0.0,
                   help='Linearly ramp the LR down to --lr-cooldown-min-ratio over the '
                        'final fraction of the run (0 disables). inverse-sqrt never '
                        'anneals -- it ends at (warmup/max_steps) ** alpha with real LR '
                        'still on, and that tail is what turns a run continuing past its '
                        'optimum into memorisation. Applied on top of either scheduler.')
    p.add_argument('--lr-cooldown-min-ratio', type=float, default=0.0,
                   help='Multiplier the cooldown ramps down to (0 = all the way off)')
    p.add_argument('--lr-batch-scaling', choices=list(LR_BATCH_SCALING_MODES),
                   default='none',
                   help='Scale --max-learning-rate by batch_size / '
                        '--lr-reference-batch-size. The objective is a batch *mean*, so '
                        'at a fixed epoch budget a run at batch B takes 1/B as many '
                        'steps of the same size: unscaled, "same epochs" quietly means '
                        '"B times less optimisation". Total drift and the gradient-noise '
                        'scale both go as lr/B, so "linear" matches both at once; '
                        '"sqrt" is the cautious compromise.')
    p.add_argument('--lr-reference-batch-size', type=int, default=1,
                   help='Batch size --max-learning-rate is quoted at (see '
                        '--lr-batch-scaling). The default of 1 leaves batch-1 runs alone')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--cpu-cores', type=int, default=1,
                   help='torch intra-op thread count (ignored on GPU)')
    p.add_argument('--device', default='cpu',
                   help="'cpu', 'cuda', 'cuda:N', or 'auto' (cuda if available, else "
                        "cpu). The fused-kernel backend (--kernels) only ever runs on "
                        "CPU float32 -- lut_lookup falls back to the plain torch path "
                        "automatically on GPU, so --kernels is simply ignored there. "
                        "table_grad's dense path (index_add_) accumulates with atomics "
                        "on CUDA, so gradients are not bit-reproducible run to run; the "
                        "parity suite is CPU-only and unaffected.")
    p.add_argument('--cuda-graphs', choices=['auto', 'on', 'off'], default='auto',
                   help="Replay forward+backward from a single CUDA-graph launch instead "
                        "of ~29,000 kernel launches (3.4x measured at the production POS "
                        "shape, batch 1; see lut_torch/graphs.py). 'auto' enables it on "
                        "CUDA and is a no-op on CPU. Numerically identical -- capture "
                        "changes which CPU work happens, not which kernels run. Requires "
                        "--table-grad dense (sparse COO tensors cannot be captured), "
                        "which is selected automatically unless you asked for sparse.")

    # validation / logging / checkpoints
    p.add_argument('--testing-length', type=int, default=10000,
                   help='Validation snippets per evaluation')
    p.add_argument('--validation-interval', type=int, default=10000)
    p.add_argument('--lm-validate-last-position-only', action='store_true',
                   help="Score only the final position of each validation window for LM, "
                        "as main.py's StandardOutputHead.validation_loss does. The default "
                        "is now every position: the forward already computes all of them, "
                        "and one per window leaves far too few scored tokens to separate "
                        "two runs. The two modes are not comparable with each other -- the "
                        "last position is the only one with a full context.")
    p.add_argument('--best-checkpoint-metric', choices=['auto', 'acc', 'loss'],
                   default='auto',
                   help="Which validation metric selects the best checkpoint. 'auto' is "
                        "accuracy for POS and loss for LM.")
    p.add_argument('--validation-interval-epochs', type=float, default=None,
                   help='Validate every this many epochs instead of every N steps. A '
                        'fixed step interval puts validation points at a different '
                        'amount of data for every batch size, so the curves cannot be '
                        'overlaid; an epoch interval lands them on identical epoch '
                        'values. Overrides --validation-interval.')
    p.add_argument('--validation-batch-size', type=int, default=0,
                   help='0 uses --batch-size')
    p.add_argument('--loss-file', type=str, default='loss.csv')
    p.add_argument('--save-model', type=str, default=None,
                   help='Checkpoint path; .pt is appended if missing')
    p.add_argument('--checkpoint-interval-validations', type=int, default=1,
                   help='Write the periodic (resume) checkpoint every Nth validation '
                        'rather than every one. The largest models here are 7.1 GB, and '
                        'validating every 0.1 epochs over a 10-epoch budget means 100 '
                        'writes -- roughly an hour of pure I/O on exactly the runs that '
                        'are already the most expensive. The best-accuracy checkpoint is '
                        'unaffected (it is written only when accuracy improves), and the '
                        'final checkpoint is always written, so this trades resume '
                        'granularity for time and nothing else.')
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
    p.add_argument('--wandb-log-points', type=int, default=0,
                   help='Target number of train-metric points for the whole run '
                        '(0 keeps --wandb-log-interval). Equalises curve density across '
                        "batch sizes: W&B's smoothing window is measured in points, so "
                        'runs logged at different densities smooth by different amounts '
                        'and read as further apart than they are.')
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
    if args.epochs is not None and args.epochs <= 0.0:
        parser.error('--epochs must be positive')
    if args.warmup_frac is not None and not 0.0 < args.warmup_frac < 1.0:
        parser.error('--warmup-frac must be in (0, 1)')
    if args.warmup_min_steps < 1:
        parser.error('--warmup-min-steps must be at least 1')
    if not 0.0 <= args.lr_cooldown_frac <= 1.0:
        parser.error('--lr-cooldown-frac must be in [0, 1]')
    if not 0.0 <= args.lr_cooldown_min_ratio <= 1.0:
        parser.error('--lr-cooldown-min-ratio must be in [0, 1]')
    if args.lr_reference_batch_size < 1:
        parser.error('--lr-reference-batch-size must be at least 1')
    if args.output_code_lr_scale <= 0.0:
        parser.error('--output-code-lr-scale must be positive')
    if (args.validation_interval_epochs is not None
            and args.validation_interval_epochs <= 0.0):
        parser.error('--validation-interval-epochs must be positive')
    if args.wandb_log_points < 0:
        parser.error('--wandb-log-points must be non-negative')
    if args.checkpoint_interval_validations < 1:
        parser.error('--checkpoint-interval-validations must be at least 1')
    if args.wandb_log_interval < 1:
        parser.error('--wandb-log-interval must be at least 1')
    if args.validation_interval < 1:
        parser.error('--validation-interval must be at least 1')
    if args.momentum and args.optimizer != 'sgd':
        parser.error('--momentum applies to --optimizer sgd')
    if args.optimizer in ('adam', 'adamw') and args.table_grad == 'auto':
        print("note: --optimizer adam/adamw requires dense table gradients; "
              "--table-grad dense will be used")


def resolve_device(args, parser):
    """``--device`` -> a ``torch.device``, resolving 'auto' and validating availability."""
    requested = args.device
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error(f"--device {args.device} requested but torch.cuda.is_available() "
                     f"is False")
    return device


def resolve_cuda_graphs(args, device, parser):
    """Whether to capture the step into a CUDA graph, forcing the settings it needs.

    Must run before :class:`LutConfig` is built, because enabling graphs pins
    ``table_grad`` to dense and the config carries that into the model.
    """
    if args.cuda_graphs == 'off':
        return False
    if device.type != 'cuda':
        if args.cuda_graphs == 'on':
            parser.error('--cuda-graphs on requires --device cuda')
        return False
    if args.table_grad == TABLE_GRAD_SPARSE:
        # Sparse COO gradients allocate per step and cannot be captured. Refuse rather
        # than silently overriding an explicit request.
        if args.cuda_graphs == 'on':
            parser.error('--cuda-graphs on is incompatible with --table-grad sparse')
        return False
    args.table_grad = TABLE_GRAD_DENSE
    return True


# ---------------------------------------------------------------------------
# Schedule and optimizers
# ---------------------------------------------------------------------------

def lr_lambda_factory(args):
    """Multiplier form of ``main.get_learning_rate`` (base LR is the peak LR).

    With ``--lr-cooldown-frac 0`` this is exactly ``main.get_learning_rate`` and the
    parity test in ``test_torch_training.py`` holds. A non-zero cooldown multiplies the
    tail by a linear ramp, which is the one thing inverse-sqrt cannot do on its own: it
    approaches ``(warmup / max_steps) ** alpha`` and stops there, leaving real LR on for
    however long the run continues past its optimum.
    """
    cooldown_steps = args.lr_cooldown_frac * args.max_steps
    # t0 == warmup is the historical schedule: the min() never binds there, because
    # (warmup/s)**alpha <= 1 for every s >= warmup, so this stays bit-identical by
    # default and only opens a plateau when a later decay start is asked for.
    decay_start = args.lr_decay_start or args.warmup_steps

    def factor(step):
        if step < args.warmup_steps:
            base = step / args.warmup_steps
        elif args.lr_scheduler == 'inverse-sqrt':
            base = min(1.0, (decay_start / step) ** args.inverse_sqrt_alpha)
        else:
            decay_steps = max(1, args.max_steps - args.warmup_steps)
            progress = min(1.0, (step - args.warmup_steps) / decay_steps)
            base = (args.cosine_min_lr_ratio + 0.5 * (1.0 - args.cosine_min_lr_ratio)
                    * (1.0 + math.cos(math.pi * progress)))
        if cooldown_steps <= 0.0:
            return base
        remaining = max(0.0, args.max_steps - step)
        ramp = min(1.0, remaining / cooldown_steps)
        return base * (args.lr_cooldown_min_ratio
                       + (1.0 - args.lr_cooldown_min_ratio) * ramp)
    return factor


def derive_schedule(args, train_set, parser):
    """Resolve the epoch budget into steps, and re-pace what used to be step-based.

    ``--max-steps`` is not a budget you can compare across batch sizes. The objective is
    a batch mean, so 200k steps is 3 passes over this corpus at batch 1 and 96 at batch
    32 -- and, at the same LR, 32x the total parameter travel to go with it. Everything
    here puts the runs back on one axis (data seen) and measures warmup, validation and
    logging against that axis instead of against the step counter.

    Called after ``load_datasets`` because every derivation needs the corpus length, and
    before ``build_optimizers`` because it is what sets the peak LR.
    """
    args.train_positions = train_set.length
    args.tokens_per_step = args.batch_size * args.context_size
    args.steps_per_epoch = args.train_positions / args.tokens_per_step

    if args.epochs is not None:
        args.max_steps = max(1, math.ceil(args.epochs * args.steps_per_epoch))
    args.epochs_run = args.max_steps / args.steps_per_epoch

    if args.warmup_frac is not None:
        args.warmup_steps = max(1, min(
            args.max_steps,
            max(args.warmup_min_steps, int(round(args.warmup_frac * args.max_steps)))))

    args.base_learning_rate = args.max_learning_rate
    args.lr_batch_scale = 1.0
    if args.lr_batch_scaling != 'none':
        ratio = args.batch_size / args.lr_reference_batch_size
        args.lr_batch_scale = (ratio if args.lr_batch_scaling == 'linear'
                               else math.sqrt(ratio))
        args.max_learning_rate = args.base_learning_rate * args.lr_batch_scale

    # Where the schedule lands before any cooldown. This is equal across batch sizes
    # only while warmup tracks max_steps; --warmup-min-steps deliberately breaks that on
    # short runs, so record the number rather than leaving it implicit in a run's config.
    if args.lr_decay_start is not None and args.lr_decay_start < args.warmup_steps:
        parser.error(f'--lr-decay-start ({args.lr_decay_start:,}) must be at least the '
                     f'resolved warmup ({args.warmup_steps:,} steps); starting the decay '
                     f'inside the ramp would step the LR down discontinuously')
    args.lr_decay_start_resolved = args.lr_decay_start or args.warmup_steps
    args.lr_decay_depth = (
        (args.lr_decay_start_resolved / args.max_steps) ** args.inverse_sqrt_alpha
        if args.lr_scheduler == 'inverse-sqrt' else args.cosine_min_lr_ratio)

    if args.validation_interval_epochs is not None:
        args.validation_interval = max(
            1, int(round(args.validation_interval_epochs * args.steps_per_epoch)))
    if args.wandb_log_points:
        args.wandb_log_interval = max(1, args.max_steps // args.wandb_log_points)

    if args.warmup_steps >= args.max_steps:
        parser.error(f'warmup ({args.warmup_steps:,} steps) must be shorter than the '
                     f'run ({args.max_steps:,} steps)')


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
    # The coded head's C and b ride in the dense optimizer but get their own param group,
    # so --output-code-lr-scale changes their step without touching PE or LayerNorm. The
    # scheduler scales each group off its own initial lr, so the shape of the schedule is
    # shared and only the peak differs. At scale 1.0 this is arithmetically identical to
    # leaving them in the dense group -- which is what keeps the already-finished
    # baseline / learned:5 / binary:5 runs comparable with anything launched after it.
    code = model.output_code_parameters()
    # Tree weights start at 1, so weight decay (which pulls towards 0) would be a prior
    # against the very model they initialise to. Same lr as everything else for now.
    tree_w = model.tree_weight_parameters()
    lr = args.max_learning_rate
    code_lr = lr * args.output_code_lr_scale
    optimizers = []

    if args.optimizer == 'sgd':
        # Plain SGD accepts either representation, so 'auto' can vary per table.
        optimizers.append(torch.optim.SGD([{'params': tables}], lr=lr))
        groups = ([{'params': dense, 'lr': lr}] if dense else [])
        if code:
            groups.append({'params': code, 'lr': code_lr})
        if tree_w:
            groups.append({'params': tree_w, 'lr': lr, 'weight_decay': 0.0})
        if groups:
            optimizers.append(torch.optim.SGD(
                groups, lr=lr, momentum=args.momentum,
                weight_decay=args.weight_decay))
    else:
        cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
        if args.table_grad == TABLE_GRAD_SPARSE:
            optimizers.append(torch.optim.SparseAdam(tables, lr=lr))
        else:
            # Adam cannot consume a sparse gradient, so pin the tables to dense.
            optimizers.append(cls(tables, lr=lr, weight_decay=args.weight_decay))
        groups = ([{'params': dense, 'lr': lr}] if dense else [])
        if code:
            groups.append({'params': code, 'lr': code_lr})
        if tree_w:
            groups.append({'params': tree_w, 'lr': lr, 'weight_decay': 0.0})
        if groups:
            optimizers.append(cls(groups, lr=lr, weight_decay=args.weight_decay))

    factor = lr_lambda_factory(args)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, factor) for opt in optimizers]
    return optimizers, schedulers


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_path(path):
    return path if path.endswith('.pt') else path + '.pt'


def best_model_path(path):
    """Sibling of ``checkpoint_path`` for the best-val-accuracy copy."""
    return (path[:-3] if path.endswith('.pt') else path) + '_best'


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
            # CUDA RNG feeds dropout on GPU; harmless (and unused) to save on a CPU run.
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available()
            else None,
        },
    }, checkpoint_path(path))


def _restore_rng(rng):
    if not rng:
        return
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng.get('torch_cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['torch_cuda'])


def load_checkpoint(path, model, optimizers, schedulers, reset_optimizer):
    ckpt = torch.load(checkpoint_path(path), map_location='cpu', weights_only=False)
    state = ckpt['model']
    converted = looks_unstacked(state, model)
    if converted:
        # A run started before --stack-heads (or with it off) wrote one table per head;
        # fold them into this model's slabs so the run can be resumed rather than
        # restarted. Optimizer state is dropped below: it is indexed by parameter, and
        # 4*H tables have just become 3.
        state = stack_state_dict(state, model)
        print('Converted a per-head checkpoint into the stacked head layout')
    model.load_state_dict(state)
    if converted and not reset_optimizer:
        print('  optimizer state discarded (parameter grouping changed); '
              'step, schedule and RNG are kept')
        for sched, sched_state in zip(schedulers, ckpt['schedulers']):
            sched.load_state_dict(sched_state)
        _restore_rng(ckpt.get('rng'))
        step = int(ckpt['step'])
        print(f"Resumed from {checkpoint_path(path)} at step {step}")
        return step
    if reset_optimizer:
        print(f"Loaded weights from {checkpoint_path(path)} (optimizer state reset)")
        return 0
    for opt, state in zip(optimizers, ckpt['optimizers']):
        opt.load_state_dict(state)
    for sched, sched_state in zip(schedulers, ckpt['schedulers']):
        sched.load_state_dict(sched_state)
    _restore_rng(ckpt.get('rng'))
    step = int(ckpt['step'])
    print(f"Resumed from {checkpoint_path(path)} at step {step}")
    return step


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def batch_loss(model, inputs, targets, vocab_size):
    """Return ``(objective, mean_loss, correct, count)`` for one batch.

    ``inputs``/``targets`` must already be on the model's device -- moved by the caller,
    once, rather than here, so a caller batching multiple chunks does not repeat the
    host-to-device copy.
    """
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
def evaluate(model, dataset, starts, args, cfg, device):
    """Validation loss / accuracy, as a dict.

    Returns ``loss``/``acc`` over every scored target plus ``loss_no_unk``/``acc_no_unk``
    over the same targets excluding ``<unk>`` (NaN for POS and for BPE tokenizers, which
    have no ``<unk>``: id 0 is a real token there), and the token counts behind each.

    For BPE tokenizers it also returns ``loss_per_word`` and ``words``: the summed NLL of
    every scored target divided by the number of scored targets that *start* a
    whitespace-delimited word (``dataset.word_tables``; NaN/0 for the word tokenizer).
    This is the standard tokenizer-independent conversion: the total NLL of a text does not
    depend on how it is cut into tokens, so dividing by words instead of tokens gives a
    per-word perplexity comparable to word-level models. The numerator includes every
    scored token (whitespace-only ones such as "\\n" too); a token whose word began before
    the window is still in the numerator but not the denominator, which is exact for the
    scored targets, since the preceding token is always ``inputs`` at the same position.

    POS scores every valid position. Language modelling scores every position too unless
    ``--lm-validate-last-position-only`` is set, which restores ``main.py``'s
    ``StandardOutputHead.validation_loss`` behaviour.

    Scoring one position per window throws away 31 of the 32 predictions the forward has
    already computed, and at ``--testing-length 1000`` leaves 1,000 scored tokens -- an
    accuracy interval wider than most differences these runs are launched to measure.
    The flag exists because the two are not comparable: the last position is the only one
    with a full context, so its loss is systematically lower than the all-position mean,
    and mixing the two across runs would be worse than either choice.
    """
    model.eval()
    batch = args.validation_batch_size or args.batch_size
    loss_sum = 0.0
    correct = 0
    count = 0
    # The same three sums again over targets that are not <unk>. <unk> is the single most
    # frequent token in this corpus (6.9%), it is a legitimate LM target, and it is not
    # one the model could have learned anything about -- so it inflates accuracy and
    # deflates perplexity by an amount nobody can read off the headline number. Both are
    # reported rather than one being chosen, because that choice is reversible only while
    # both have been recorded. POS is excluded: its targets are tag ids, where id 0 is
    # ADJ, not <unk> (data.py already masks POS targets whose *input* was <unk>).
    unk_id = getattr(dataset, 'unk_id', UNK_TOKEN_ID)
    score_no_unk = not cfg.POS_task and unk_id is not None
    tables = None if cfg.POS_task else getattr(dataset, 'word_tables', None)
    words = 0
    loss_sum_nu = 0.0
    correct_nu = 0
    count_nu = 0
    for begin in tqdm(range(0, len(starts), batch), desc='  Val', leave=False,
                      unit='batch'):
        chunk = starts[begin:begin + batch]
        inputs, targets = dataset.batch(chunk)
        if not cfg.POS_task and args.lm_validate_last_position_only:
            keep = np.full_like(targets, IGNORE_INDEX)
            keep[:, -1] = targets[:, -1]
            targets = keep
        logits = model(torch.from_numpy(inputs).to(device))
        flat = logits.reshape(-1, cfg.vocab_size_output)
        flat_targets = torch.from_numpy(targets).to(device).reshape(-1)
        valid = flat_targets != IGNORE_INDEX
        if not bool(valid.any()):
            continue
        hit = flat.argmax(dim=-1) == flat_targets
        loss_sum += float(F.cross_entropy(flat[valid], flat_targets[valid],
                                          reduction='sum'))
        correct += int(hit[valid].sum())
        count += int(valid.sum())
        if tables is not None:
            starts_word = tables.starts_word(inputs, targets)
            words += int(starts_word[targets != IGNORE_INDEX].sum())
        if score_no_unk:
            keep_nu = valid & (flat_targets != unk_id)
            if bool(keep_nu.any()):
                loss_sum_nu += float(F.cross_entropy(flat[keep_nu],
                                                     flat_targets[keep_nu],
                                                     reduction='sum'))
                correct_nu += int(hit[keep_nu].sum())
                count_nu += int(keep_nu.sum())
    model.train()
    nan = float('nan')
    return {
        'loss': loss_sum / count if count else nan,
        'acc': correct / count if count else nan,
        'count': count,
        'loss_no_unk': loss_sum_nu / count_nu if count_nu else nan,
        'acc_no_unk': correct_nu / count_nu if count_nu else nan,
        'count_no_unk': count_nu,
        'loss_per_word': loss_sum / words if tables is not None and words else nan,
        'words': words,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_cores))
    set_kernel_backend(args.kernels)
    if args.kernels in ('auto', 'numba'):
        try:
            import numba
            numba.set_num_threads(max(1, args.cpu_cores))
            set_kernel_threads(max(1, args.cpu_cores))
        except ImportError:
            pass

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

    # Device and the graph decision both come before the config: enabling graphs pins
    # table_grad to dense, and LutConfig carries that setting into the model.
    device = resolve_device(args, parser)
    use_graphs = resolve_cuda_graphs(args, device, parser)
    # A single head has nothing to stack, so 'auto' leaves it alone rather than paying
    # for the reshapes that make the stacked layout work.
    args.stack_heads = (args.stack_heads_mode == 'on'
                        or (args.stack_heads_mode == 'auto' and args.num_heads > 1))

    try:
        cfg = LutConfig.from_namespace(args)
    except ConfigError as exc:
        parser.error(str(exc))

    print('=' * 50)
    print('Configuration:')
    for key, value in sorted(cfg.to_dict().items()):
        print(f"  {key:24s} = {value}")
    # max_steps / warmup / LR / validation cadence are all derived from the corpus
    # length below, so they are printed by the Schedule block rather than here.
    for key in ('batch_size', 'optimizer', 'table_grad', 'momentum', 'weight_decay',
                'lr_scheduler', 'inverse_sqrt_alpha', 'cosine_min_lr_ratio',
                'seed', 'cpu_cores', 'kernels', 'device', 'cuda_graphs',
                'testing_length', 'loss_file', 'save_model', 'load_model',
                'output_code_lr_scale', 'best_checkpoint_metric',
                'lm_validate_last_position_only'):
        print(f"  {key:24s} = {getattr(args, key)}")
    print('=' * 50)

    train_set, val_set = load_datasets(args, enc)
    derive_schedule(args, train_set, parser)

    print('Schedule:')
    for key in ('epochs_run', 'steps_per_epoch', 'tokens_per_step', 'max_steps',
                'warmup_steps', 'lr_decay_start_resolved',
                'base_learning_rate', 'lr_batch_scale',
                'max_learning_rate', 'lr_decay_depth', 'lr_cooldown_frac',
                'validation_interval', 'wandb_log_interval'):
        value = getattr(args, key)
        print(f"  {key:24s} = " + (f"{value:.6g}" if isinstance(value, float)
                                   else f"{value:,}"))
    print('=' * 50)

    # Drawn exactly as main.load_training_data draws testing_input_data: Python's
    # `random`, one inclusive randint per snippet. Two things depend on this. The
    # validation windows themselves match main.py's at the same seed, so a torch run and
    # a main.py run report accuracy over the *same* snippets -- with only a few thousand
    # scored tokens, two independently drawn window sets differ in difficulty by more
    # than the model differences usually under test. And because this consumes Python
    # RNG rather than NumPy RNG, both streams are left where main.build_Model finds
    # them, so construction below matches main.py's too.
    val_starts = np.array([random.randint(0, val_set.length - 1)
                           for _ in range(args.testing_length)], dtype=np.int64)

    model = LutModel(cfg).to(device)
    print(f"Trainable tables: {len(model.table_parameters())}, "
          f"table entries: {sum(p.numel() for p in model.table_parameters()):,}, "
          f"dense parameters: {sum(p.numel() for p in model.dense_parameters()):,}, "
          f"output code parameters: "
          f"{sum(p.numel() for p in model.output_code_parameters()):,} "
          f"@ lr x{args.output_code_lr_scale:g}, "
          f"tree weights: {sum(p.numel() for p in model.tree_weight_parameters()):,}, "
          f"device: {device}, cuda graphs: {'on' if use_graphs else 'off'}")

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

    def epoch_at(s):
        return s * args.tokens_per_step / args.train_positions

    # The final checkpoint is not the best one. Under a fixed-step schedule these runs
    # peaked as early as 18% in and then trained for another ten hours, so what landed
    # on disk was never the model the run actually found; keep both.
    best = {'acc': float('-inf'), 'loss': float('inf'), 'step': -1, 'epoch': 0.0}
    validations = 0

    with open(args.loss_file, 'w') as f:
        # The no_unk triple is word-tokenizer LM only and the per_word pair is BPE LM
        # only; both are NaN otherwise. See evaluate().
        f.write('step, epoch, tokens, loss, perplexity, accuracy, '
                'loss_no_unk, perplexity_no_unk, accuracy_no_unk, '
                'loss_per_word, perplexity_per_word\n')

    model.train()
    ema_loss = None
    ema_alpha = 0.02
    last = {}

    # The graph is captured on the first real batch, after any checkpoint has been
    # loaded, so it records the shapes and parameters the run actually uses.
    graphed = None

    pbar = tqdm(range(start_step, args.max_steps), desc='Training', unit='step')
    for step in pbar:
        starts = np.random.randint(0, train_set.length, size=args.batch_size)
        inputs, targets = train_set.batch(starts)
        inputs = torch.from_numpy(inputs).to(device)
        targets = torch.from_numpy(targets).to(device)

        if use_graphs:
            if graphed is None:
                graphed = GraphedTrainStep(model, inputs, targets, cfg.vocab_size_output,
                                           IGNORE_INDEX, args.batch_size)
                tqdm.write('Captured forward+backward into a CUDA graph')
            # No zero_grad: every replay overwrites .grad in place (see graphs.py).
            mean_loss, _, _ = graphed.run(inputs, targets)
            if step == start_step + 1:
                # One check, after a full replay + opt.step() cycle has run: catches a
                # future edit that reintroduces zero_grad(set_to_none=True) here, which
                # would otherwise show up only as a run that trains at full speed and
                # learns nothing.
                graphed.assert_grads_intact()
        else:
            objective, mean_loss, _, _ = batch_loss(model, inputs, targets,
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
            # tqdm's own smoothed rate, i.e. exactly the it/s shown on the bar. It
            # is None on the very first iteration, before any interval is timed.
            rate = pbar.format_dict.get('rate')
            payload = {
                'step': step,
                'epoch': epoch_at(step),
                'tokens': step * args.tokens_per_step,
                'train/loss': mean_loss,
                'train/ema_loss': ema_loss,
                'train/perplexity': math.exp(min(ema_loss, 20.0)),
                'train/learning_rate': lr,
            }
            if rate:
                payload['train/steps_per_sec'] = rate
                payload['train/tokens_per_sec'] = rate * args.tokens_per_step
            wandb_run.log(payload)

        if step % args.validation_interval == 0:
            pbar.set_description('Validating')
            val = evaluate(model, val_set, val_starts, args, cfg, device)
            val_loss, val_acc = val['loss'], val['acc']
            perplexity = math.exp(min(val_loss, 20.0)) if val_loss == val_loss \
                else float('nan')
            ppl_no_unk = (math.exp(min(val['loss_no_unk'], 20.0))
                          if val['loss_no_unk'] == val['loss_no_unk'] else float('nan'))
            ppl_word = (math.exp(min(val['loss_per_word'], 20.0))
                        if val['loss_per_word'] == val['loss_per_word'] else float('nan'))
            last = {'loss': val_loss, 'ppl': perplexity, 'acc': val_acc}

            with open(args.loss_file, 'a') as f:
                f.write(f"{step}, {epoch_at(step):.4f}, {step * args.tokens_per_step}, "
                        f"{val_loss:.6f}, {perplexity:.2f}, {val_acc:.6f}, "
                        f"{val['loss_no_unk']:.6f}, {ppl_no_unk:.2f}, "
                        f"{val['acc_no_unk']:.6f}, "
                        f"{val['loss_per_word']:.6f}, {ppl_word:.2f}\n")
            line = (f"\n--- step {step:,} | ppl={perplexity:.2f} | "
                    f"loss={val_loss:.3f} | acc={val_acc:.3f}")
            if val['count_no_unk']:
                line += (f" | no-unk ppl={ppl_no_unk:.2f} "
                         f"acc={val['acc_no_unk']:.3f}")
            if val['words']:
                line += f" | ppl/word={ppl_word:.2f}"
            tqdm.write(line + ' ---')

            validations += 1
            # POS selects on accuracy, which is the metric it is judged by. LM selects on
            # validation loss: perplexity is the LM metric, loss is the more sensitive
            # signal (DECISIONS.md), and early LM accuracy improves at nearly every
            # validation -- which on a full-vocabulary model means writing a ~1.1 GB
            # checkpoint each time.
            if args.best_checkpoint_metric == 'auto':
                metric = 'acc' if cfg.POS_task else 'loss'
            else:
                metric = args.best_checkpoint_metric
            if metric == 'acc':
                improved = val_acc == val_acc and val_acc > best['acc']
            else:
                improved = val_loss == val_loss and val_loss < best['loss']
            if improved:
                best.update(acc=val_acc, loss=val_loss, step=step, epoch=epoch_at(step))

            if args.save_model:
                if validations % args.checkpoint_interval_validations == 0:
                    save_checkpoint(args.save_model, model, cfg, optimizers, schedulers,
                                    step, args)
                    tqdm.write(f"Saved checkpoint to {checkpoint_path(args.save_model)}")
                if improved:
                    save_checkpoint(best_model_path(args.save_model), model, cfg,
                                    optimizers, schedulers, step, args)
                    shown = val_acc if metric == 'acc' else val_loss
                    tqdm.write(f"  new best val {metric} {shown:.4f} -> "
                               f"{checkpoint_path(best_model_path(args.save_model))}")
            if wandb_run is not None:
                wandb_run.log({'step': step, 'epoch': epoch_at(step),
                               'tokens': step * args.tokens_per_step,
                               'val/loss': val_loss, 'val/perplexity': perplexity,
                               'val/accuracy': val_acc,
                               'val/loss_no_unk': val['loss_no_unk'],
                               'val/perplexity_no_unk': ppl_no_unk,
                               'val/accuracy_no_unk': val['acc_no_unk'],
                               'val/scored_tokens': val['count'],
                               'val/scored_tokens_no_unk': val['count_no_unk'],
                               'val/loss_per_word': val['loss_per_word'],
                               'val/perplexity_per_word': ppl_word,
                               'val/scored_words': val['words']})
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
    if best['step'] >= 0:
        print(f"Best val accuracy {best['acc']:.4f} at step {best['step']:,} "
              f"(epoch {best['epoch']:.2f}), val loss {best['loss']:.4f}")
    if wandb_run is not None:
        if best['step'] >= 0:
            # In the summary, so a sweep table can be sorted by peak rather than by
            # whatever the run happened to end on.
            wandb_run.summary['val/best_acc'] = best['acc']
            wandb_run.summary['val/best_loss'] = best['loss']
            wandb_run.summary['val/best_step'] = best['step']
            wandb_run.summary['val/best_epoch'] = best['epoch']
        wandb_run.finish()


if __name__ == '__main__':
    main()
