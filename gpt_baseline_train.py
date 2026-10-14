"""nanoGPT language-model baseline on exactly the LUT LM's data, schedule and metrics.

    python gpt_baseline_train.py TRAIN_FILE --validation-data VAL_FILE \
        --vocab-file VOCAB --n-layer 6 --n-head 4 --n-embd 64 --epochs 30 ...

The model is ``nanoGPT_POS/model.py``'s ``GPT``, unmodified. Everything around it is
borrowed from ``lut_torch.train`` rather than reimplemented, so the only thing that
differs from a ``lut_torch`` LM run is the network:

* data: ``lut_torch.data.load_datasets`` -- the same ``WordTokenizer`` (or tiktoken, with
  ``--tokenizer gpt2``, text or pre-tokenized ``.bin``) stream and the same
  ``SnippetDataset`` windows (inputs ``tokens[s:s+CS]``, targets ``tokens[s+1:s+CS+1]``);
* validation windows: drawn with Python ``random`` right after seeding, exactly as
  ``lut_torch.train.main`` draws them, so at the same ``--seed`` and ``--testing-length``
  this run and a ``lut_torch`` run score the *same* snippets (asserted in
  ``test_gpt_baseline.py``) and their ``loss.csv`` files are directly comparable;
* metrics: ``lut_torch.train.evaluate`` -- every position scored, ``<unk>`` counted as a
  target, plus the ``_no_unk`` scoring; same 11-column ``loss.csv`` (the ``_no_unk`` triple is NaN under BPE, the
  ``_per_word`` pair NaN under the word tokenizer);
* schedule: ``lut_torch.train.derive_schedule`` and ``lr_lambda_factory`` -- an
  ``--epochs`` budget, ``--warmup-frac``, cosine to ``--cosine-min-lr-ratio``.

What is deliberately *not* shared is the optimizer: the transformer gets the usual
nanoGPT recipe (AdamW, betas 0.9/0.95, weight decay on 2-D weights, grad clip 1.0, bf16
autocast on CUDA), because a baseline trained with the LUT's SGD would be a strawman.

LM only: a POS baseline already exists (``nanoGPT_POS/train.py``). No resume: runs at
these sizes take minutes to a few hours, so a preempted run is simply relaunched.
"""

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from lut_torch.data import IGNORE_INDEX, load_datasets  # noqa: E402
from lut_torch.train import (  # noqa: E402
    DEFAULT_COSINE_MIN_LR_RATIO,
    LR_BATCH_SCALING_MODES,
    derive_schedule,
    evaluate,
    lr_lambda_factory,
)


def load_nanogpt():
    """Import ``nanoGPT_POS/model.py`` under a unique name (``model`` is too generic)."""
    path = os.path.join(REPO, 'nanoGPT_POS', 'model.py')
    spec = importlib.util.spec_from_file_location('nanogpt_model', path)
    module = importlib.util.module_from_spec(spec)
    # torch.compile re-imports the module by name when tracing its globals.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FullLogits(nn.Module):
    """``GPT`` returning ``(B, T, V)`` logits at every position, as ``evaluate`` expects.

    ``GPT.forward`` without targets keeps only the last position (a sampling shortcut),
    and with targets returns a mean loss we do not want. This is the same forward with
    neither shortcut.
    """

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, idx):
        g = self.gpt.transformer
        pos = torch.arange(idx.size(1), dtype=torch.long, device=idx.device)
        x = g.drop(g.wte(idx) + g.wpe(pos))
        for block in g.h:
            x = block(x)
        return self.gpt.lm_head(g.ln_f(x)).float()


def draw_val_starts(val_set, testing_length):
    """``lut_torch.train.main``'s draw, verbatim. Must run right after seeding."""
    return np.array([random.randint(0, val_set.length - 1)
                     for _ in range(testing_length)], dtype=np.int64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('training_data')
    p.add_argument('--validation-data', required=True)
    p.add_argument('--tokenizer', default='word',
                   help="'word' (needs --vocab-file) or a tiktoken encoding name such as "
                        "gpt2, as in lut_torch.")
    p.add_argument('--vocab-file', default=None, help='Required for --tokenizer word.')
    p.add_argument('--context-size', type=int, default=32)

    g = p.add_argument_group('model (nanoGPT GPTConfig)')
    g.add_argument('--n-layer', type=int, default=6)
    g.add_argument('--n-head', type=int, default=4)
    g.add_argument('--n-embd', type=int, default=64)
    g.add_argument('--dropout', type=float, default=0.2)
    g.add_argument('--bias', action='store_true',
                   help='Biases in Linear/LayerNorm (nanoGPT default here: off).')

    g = p.add_argument_group('optimization')
    g.add_argument('--batch-size', type=int, default=32)
    g.add_argument('--epochs', type=float, default=None)
    g.add_argument('--max-steps', type=int, default=None)
    g.add_argument('--warmup-steps', type=int, default=200)
    g.add_argument('--warmup-frac', type=float, default=None)
    g.add_argument('--warmup-min-steps', type=int, default=200)
    g.add_argument('--max-learning-rate', type=float, default=1e-3)
    g.add_argument('--lr-batch-scaling', choices=LR_BATCH_SCALING_MODES, default='none')
    g.add_argument('--lr-reference-batch-size', type=int, default=1)
    g.add_argument('--lr-scheduler', choices=('cosine', 'inverse-sqrt'), default='cosine')
    g.add_argument('--cosine-min-lr-ratio', type=float,
                   default=DEFAULT_COSINE_MIN_LR_RATIO)
    g.add_argument('--weight-decay', type=float, default=0.1)
    g.add_argument('--beta1', type=float, default=0.9)
    g.add_argument('--beta2', type=float, default=0.95)
    g.add_argument('--grad-clip', type=float, default=1.0, help='0 disables.')
    g.add_argument('--seed', type=int, default=42)

    g = p.add_argument_group('runtime')
    g.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:N')
    g.add_argument('--dtype', choices=('auto', 'float32', 'bfloat16', 'float16'),
                   default='auto',
                   help='Autocast dtype for the *training* forward; validation is '
                        'always float32. auto = bfloat16 on CUDA, float32 on CPU.')
    g.add_argument('--compile', choices=('auto', 'on', 'off'), default='auto',
                   help='torch.compile the model (auto = on for CUDA).')
    g.add_argument('--cpu-cores', type=int, default=4)

    g = p.add_argument_group('validation, checkpoints, logging')
    g.add_argument('--testing-length', type=int, default=1000)
    g.add_argument('--validation-batch-size', type=int, default=100)
    g.add_argument('--validation-interval', type=int, default=1000)
    g.add_argument('--validation-interval-epochs', type=float, default=None)
    g.add_argument('--loss-file', default='loss.csv')
    g.add_argument('--save-model', default=None,
                   help='Path stem; writes STEM.pt (final) and STEM_best.pt (best val '
                        'loss).')
    g.add_argument('--wandb', action='store_true')
    g.add_argument('--wandb-project', default='LUT-LM-gpt-baseline')
    g.add_argument('--wandb-entity', default=None)
    g.add_argument('--wandb-run-name', default=None)
    g.add_argument('--wandb-group', default=None)
    g.add_argument('--wandb-tags', nargs='*', default=None)
    g.add_argument('--wandb-mode', default='online')
    g.add_argument('--wandb-log-interval', type=int, default=100)
    g.add_argument('--wandb-log-points', type=int, default=0)
    return p


def validate_args(args, parser):
    if args.n_embd % args.n_head:
        parser.error('--n-embd must be divisible by --n-head')
    if (args.epochs is None) == (args.max_steps is None):
        parser.error('give exactly one of --epochs / --max-steps')
    if not 0.0 <= args.dropout < 1.0:
        parser.error('--dropout must be in [0, 1)')
    if args.testing_length < 1:
        parser.error('--testing-length must be at least 1')
    # derive_schedule / lr_lambda_factory read these; the LUT trainer's cooldown and
    # decay-start knobs are not part of the transformer recipe, so they are pinned off.
    args.lr_cooldown_frac = 0.0
    args.lr_cooldown_min_ratio = 0.0
    args.lr_decay_start = None
    args.inverse_sqrt_alpha = 0.5
    # evaluate() reads these.
    args.lm_validate_last_position_only = False
    # load_datasets() reads this.
    args.POS_task = False
    if args.tokenizer == 'word' and not args.vocab_file:
        parser.error('--vocab-file is required for --tokenizer word')


def resolve_runtime(args):
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    if args.dtype == 'auto':
        args.dtype = 'bfloat16' if device.type == 'cuda' else 'float32'
    if args.compile == 'auto':
        args.compile = 'on' if device.type == 'cuda' else 'off'
    return device


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_path(stem):
    return stem if stem.endswith('.pt') else stem + '.pt'


def best_model_path(stem):
    return (stem[:-3] if stem.endswith('.pt') else stem) + '_best'


def save_checkpoint(stem, gpt, model_args, optimizer, step, val, args):
    torch.save({
        'kind': 'nanogpt_lm',
        'step': step,
        'model_args': model_args,
        'model': gpt.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val': val,
        'args': {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, bool, str, type(None), list))},
    }, checkpoint_path(stem))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_cores))
    device = resolve_runtime(args)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.tokenizer == 'word':
        from word_tokenizer import WordTokenizer
        enc = WordTokenizer(args.vocab_file)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(args.tokenizer)
    vocab_size = enc.n_vocab    # 50257 for gpt2; nanoGPT's tied embedding handles it

    train_set, val_set = load_datasets(args, enc)
    # Immediately after load_datasets, with no Python-RNG use in between: this is the
    # position lut_torch.train draws them from, which is what makes the windows equal.
    val_starts = draw_val_starts(val_set, args.testing_length)
    derive_schedule(args, train_set, parser)

    nanogpt = load_nanogpt()
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      block_size=args.context_size, bias=args.bias,
                      vocab_size=vocab_size, dropout=args.dropout)
    gpt = nanogpt.GPT(nanogpt.GPTConfig(**model_args)).to(device)
    model = FullLogits(gpt)
    n_params = sum(p.numel() for p in gpt.parameters())
    n_non_emb = n_params - gpt.transformer.wpe.weight.numel() \
        - gpt.transformer.wte.weight.numel()
    args.num_params = n_params
    args.num_params_non_embedding = n_non_emb

    optimizer = gpt.configure_optimizers(args.weight_decay, args.max_learning_rate,
                                         (args.beta1, args.beta2), device.type)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(args))

    print('=' * 50)
    print('Configuration:')
    for key, value in sorted(model_args.items()):
        print(f"  {key:24s} = {value}")
    for key in ('batch_size', 'max_learning_rate', 'lr_scheduler',
                'cosine_min_lr_ratio', 'weight_decay', 'beta1', 'beta2', 'grad_clip',
                'seed', 'device', 'dtype', 'compile', 'testing_length', 'loss_file',
                'save_model'):
        print(f"  {key:24s} = {getattr(args, key)}")
    print('Schedule:')
    for key in ('epochs_run', 'steps_per_epoch', 'tokens_per_step', 'max_steps',
                'warmup_steps', 'validation_interval'):
        value = getattr(args, key)
        print(f"  {key:24s} = " + (f"{value:.6g}" if isinstance(value, float)
                                   else f"{value:,}"))
    print(f"Parameters: {n_params:,} total, {n_non_emb:,} excluding token and "
          f"position embeddings (token embedding is tied to the output head)")
    print('=' * 50)

    train_model = torch.compile(model) if args.compile == 'on' else model
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}[args.dtype]
    autocast = (nullcontext() if args.dtype == 'float32' else
                torch.autocast(device_type=device.type, dtype=ptdtype))
    scaler = torch.amp.GradScaler(device.type, enabled=(args.dtype == 'float16'))

    # evaluate() only reads these two fields of the LUT config.
    eval_cfg = SimpleNamespace(POS_task=False, vocab_size_output=vocab_size)

    wandb_run = None
    if args.wandb:
        import wandb
        config = {k: v for k, v in vars(args).items()
                  if isinstance(v, (int, float, bool, str, type(None), list, tuple))}
        config.update(model_args)
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               group=args.wandb_group, entity=args.wandb_entity,
                               tags=args.wandb_tags, mode=args.wandb_mode,
                               config=config)
        wandb.define_metric('step')
        wandb.define_metric('train/*', step_metric='step')
        wandb.define_metric('val/*', step_metric='step')

    def epoch_at(s):
        return s * args.tokens_per_step / args.train_positions

    with open(args.loss_file, 'w') as f:
        f.write('step, epoch, tokens, loss, perplexity, accuracy, '
                'loss_no_unk, perplexity_no_unk, accuracy_no_unk, '
                'loss_per_word, perplexity_per_word\n')

    best = {'loss': float('inf'), 'acc': float('nan'), 'step': -1, 'epoch': 0.0}
    last = {}
    ema_loss = None
    model.train()
    t_start = time.time()
    pbar = tqdm(range(args.max_steps), desc='Training', unit='step')
    for step in pbar:
        starts = np.random.randint(0, train_set.length, size=args.batch_size)
        inputs, targets = train_set.batch(starts)
        inputs = torch.from_numpy(inputs).to(device, non_blocking=True)
        targets = torch.from_numpy(targets).to(device, non_blocking=True)

        with autocast:
            logits = train_model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1),
                               ignore_index=IGNORE_INDEX)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        mean_loss = float(loss.detach())
        ema_loss = mean_loss if ema_loss is None else 0.02 * mean_loss + 0.98 * ema_loss
        lr = optimizer.param_groups[0]['lr']

        if wandb_run is not None and step % args.wandb_log_interval == 0:
            wandb_run.log({'step': step, 'epoch': epoch_at(step),
                           'tokens': step * args.tokens_per_step,
                           'train/loss': mean_loss, 'train/ema_loss': ema_loss,
                           'train/perplexity': math.exp(min(ema_loss, 20.0)),
                           'train/learning_rate': lr})

        # Same cadence as lut_torch (validates after step 0, then every interval), plus
        # the final step, so min and final val are both on record.
        if step % args.validation_interval == 0 or step == args.max_steps - 1:
            val = evaluate(model, val_set, val_starts, args, eval_cfg, device)
            ppl = math.exp(min(val['loss'], 20.0))
            ppl_nu = math.exp(min(val['loss_no_unk'], 20.0))
            ppl_word = math.exp(min(val['loss_per_word'], 20.0))
            last = {'ppl': ppl, 'loss': val['loss'], 'acc': val['acc']}
            with open(args.loss_file, 'a') as f:
                f.write(f"{step}, {epoch_at(step):.4f}, {step * args.tokens_per_step}, "
                        f"{val['loss']:.6f}, {ppl:.2f}, {val['acc']:.6f}, "
                        f"{val['loss_no_unk']:.6f}, {ppl_nu:.2f}, "
                        f"{val['acc_no_unk']:.6f}, "
                        f"{val['loss_per_word']:.6f}, {ppl_word:.2f}\n")
            msg = (f"\n--- step {step:,} | ppl={ppl:.2f} | loss={val['loss']:.3f} | "
                   f"acc={val['acc']:.3f}")
            if val['count_no_unk']:
                msg += f" | no-unk ppl={ppl_nu:.2f} acc={val['acc_no_unk']:.3f}"
            if val['words']:
                msg += f" | ppl/word={ppl_word:.2f}"
            tqdm.write(msg + ' ---')
            improved = val['loss'] < best['loss']
            if improved:
                best.update(loss=val['loss'], acc=val['acc'], step=step,
                            epoch=epoch_at(step))
                if args.save_model:
                    save_checkpoint(best_model_path(args.save_model), gpt, model_args,
                                    optimizer, step, val, args)
            if wandb_run is not None:
                wandb_run.log({'step': step, 'epoch': epoch_at(step),
                               'tokens': step * args.tokens_per_step,
                               'val/loss': val['loss'], 'val/perplexity': ppl,
                               'val/accuracy': val['acc'],
                               'val/loss_no_unk': val['loss_no_unk'],
                               'val/perplexity_no_unk': ppl_nu,
                               'val/accuracy_no_unk': val['acc_no_unk'],
                               'val/scored_tokens': val['count'],
                               'val/scored_tokens_no_unk': val['count_no_unk'],
                               'val/loss_per_word': val['loss_per_word'],
                               'val/perplexity_per_word': ppl_word,
                               'val/scored_words': val['words']})

        postfix = {'lr': f"{lr:.2e}", 'train_ppl': f"{math.exp(min(ema_loss, 20.0)):.1f}"}
        if last:
            postfix.update(val_ppl=f"{last['ppl']:.2f}")
        pbar.set_postfix(**postfix)

    elapsed_h = (time.time() - t_start) / 3600
    if args.save_model:
        save_checkpoint(args.save_model, gpt, model_args, optimizer, args.max_steps,
                        last, args)
        print(f"Saved final checkpoint to {checkpoint_path(args.save_model)}")
    print(f"Best val loss {best['loss']:.4f} (ppl {math.exp(min(best['loss'], 20.0)):.2f}, "
          f"acc {best['acc']:.4f}) at step {best['step']:,} (epoch {best['epoch']:.2f}); "
          f"final ppl {last.get('ppl', float('nan')):.2f}; {elapsed_h:.2f} h")
    if wandb_run is not None:
        wandb_run.summary['val/best_loss'] = best['loss']
        wandb_run.summary['val/best_perplexity'] = math.exp(min(best['loss'], 20.0))
        wandb_run.summary['val/best_acc'] = best['acc']
        wandb_run.summary['val/best_step'] = best['step']
        wandb_run.summary['val/best_epoch'] = best['epoch']
        wandb_run.summary['num_params'] = n_params
        wandb_run.finish()


if __name__ == '__main__':
    main()
