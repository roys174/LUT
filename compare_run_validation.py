"""Evaluate `main.py` (.npz) and `lut_torch` (.pt) checkpoints on one shared validation set.

Each trainer draws its own fixed validation windows from its own RNG stream, so two runs
of the same configuration report accuracy on *different* snippets of the validation file.
At `--testing-length 100` (~2,900 scored tokens) that alone moves accuracy by about
+/-1 point, which is larger than most differences worth arguing about. This script removes
that confound: it scores every checkpoint on the same pinned window starts.

A run directory is identified by its `results/**/config.txt`; the command recorded there
supplies the structural flags, so only the run directory needs naming:

    .venv/bin/python compare_run_validation.py \
        results/POS_new_attn/20260814-101642 \
        results/POS_torch_gpu/20260906-142646_torch_gpu_WS8_bs1_sgd \
        --windows 1000

`main.py` checkpoints are rebuilt through `main.main`'s own setup path (stopped at
`print_model_stats`) so construction is RNG-faithful, then overwritten by `load_model`.
Both paths run on CPU.
"""

import argparse
import os
import shlex
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
IGNORE_INDEX = -1


def read_config(run_dir):
    """Return (argv, kind) for a run directory, from its config.txt command line."""
    path = os.path.join(run_dir, 'config.txt')
    with open(path) as f:
        text = f.read()
    if 'command:' not in text:
        raise SystemExit(f'{path}: no "command:" section')
    command = text.split('command:', 1)[1].strip()
    argv = shlex.split(command)
    if argv[:1] == ['python']:
        argv = argv[1:]
    if argv[:2] == ['-m', 'lut_torch.train']:
        return argv[2:], 'torch'
    if argv[:1] == ['main.py']:
        return argv[1:], 'main'
    raise SystemExit(f'{path}: unrecognised command {argv[:3]}')


def flag_value(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def checkpoint_path(run_dir, kind):
    name = 'model.npz' if kind == 'main' else 'model.pt'
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        raise SystemExit(f'missing checkpoint {path}')
    return path


def load_validation_stream(argv, enc, pos_loader):
    """Return (tokens, pos_labels) for the run's validation file."""
    val_file = flag_value(argv, '--validation-data')
    if val_file is None:
        raise SystemExit('runs without --validation-data are not supported here')
    with open(val_file, encoding='utf-8') as f:
        tokens = np.array(enc.encode(f.read()), dtype=np.int32)
    return val_file, tokens, pos_loader(val_file, tokens)


def windows(tokens, pos, starts, context_size):
    """Return (inputs, targets) with POS targets masked wherever the input is <unk>."""
    offsets = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(context_size + 1)
    window = tokens[offsets]
    targets = pos[offsets].astype(np.int64, copy=True)
    targets[window == 0] = IGNORE_INDEX
    return window[:, :context_size].astype(np.int64), targets[:, :context_size]


# ---------------------------------------------------------------------------
# main.py checkpoints
# ---------------------------------------------------------------------------

def eval_main_checkpoint(argv, ckpt, starts):
    sys.path.insert(0, REPO)
    import main as M

    captured = {}

    class _Stop(Exception):
        pass

    def _grab(m, args):
        captured['m'], captured['args'] = m, args
        raise _Stop()

    real_stats, M.print_model_stats = M.print_model_stats, _grab
    real_argv = sys.argv
    # A run's own --loss-file/--save-model/--wandb must not be re-triggered here.
    clean = []
    skip = 0
    for i, tok in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if tok in ('--loss-file', '--save-model', '--wandb-run-name', '--wandb-project',
                   '--wandb-entity', '--wandb-group', '--wandb-tags'):
            skip = 1
            continue
        if tok == '--wandb':
            continue
        clean.append(tok)
    sys.argv = ['main.py'] + clean + ['--loss-file', os.devnull]
    try:
        M.main()
    except _Stop:
        pass
    finally:
        sys.argv = real_argv
        M.print_model_stats = real_stats

    m, args = captured['m'], captured['args']
    M.load_model(m, ckpt, args)

    val_file, tokens, pos = load_validation_stream(
        argv, args.enc,
        lambda path, toks: M.load_or_build_pos_encoding(path, toks, args, 'validation'))
    cs = args.context_size
    preds = np.empty((len(starts), cs), dtype=np.int64)
    targets = np.empty((len(starts), cs), dtype=np.int64)
    total, count = 0.0, 0
    for i, start in enumerate(starts):
        M.load_snippet(m, tokens, int(start), args, pos)
        M.model_forward(m, args, training=False)
        # argmax/targets are read before validation_loss, which softmaxes in place.
        preds[i] = np.argmax(m.output_head.output[:cs], axis=1)
        targets[i] = m.output_head.tokens[:cs]
        result = m.output_head.validation_loss(args)
        if result is not None:
            loss_sum, n = result
            total += loss_sum
            count += n
    return preds.reshape(-1), targets.reshape(-1), (total, count)


# ---------------------------------------------------------------------------
# lut_torch checkpoints
# ---------------------------------------------------------------------------

def eval_torch_checkpoint(argv, ckpt, starts, batch, threads):
    sys.path.insert(0, REPO)
    import torch
    import torch.nn.functional as F
    from lut_torch.config import LutConfig
    from lut_torch.modules import LutModel
    from word_tokenizer import WordTokenizer
    import main as M  # only for the shared POS cache loader

    torch.set_num_threads(threads)
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg_fields = {k: v for k, v in blob['config'].items()
                  if LutConfig.__dataclass_fields__[k].init}
    cfg = LutConfig(**cfg_fields)
    model = LutModel(cfg)
    model.load_state_dict(blob['model'])
    model.eval()

    enc = WordTokenizer(flag_value(argv, '--vocab-file'))
    ns = argparse.Namespace(pos_cache_dir=flag_value(argv, '--pos-cache-dir'),
                            tokenizer=flag_value(argv, '--tokenizer', 'word'),
                            vocab_file=flag_value(argv, '--vocab-file'))
    val_file, tokens, pos = load_validation_stream(
        argv, enc, lambda path, toks: M.load_or_build_pos_encoding(path, toks, ns, 'validation'))

    total, count = 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for begin in range(0, len(starts), batch):
            inputs, tgt = windows(tokens, pos, starts[begin:begin + batch],
                                  cfg.context_size)
            flat = model(torch.from_numpy(inputs)).reshape(-1, cfg.vocab_size_output)
            flat_t = torch.from_numpy(tgt).reshape(-1)
            valid = flat_t != IGNORE_INDEX
            total += float(F.cross_entropy(flat[valid], flat_t[valid], reduction='sum'))
            count += int(valid.sum())
            preds.append(flat.argmax(dim=-1).numpy())
            targets.append(flat_t.numpy())
    return np.concatenate(preds), np.concatenate(targets), (total, count)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='+', help='results/<experiment>/<run> directories')
    p.add_argument('--windows', type=int, default=1000,
                   help='validation windows to score (default: 1000)')
    p.add_argument('--window-seed', type=int, default=12345,
                   help='seed for the shared window starts (default: 12345)')
    p.add_argument('--starts-file', default=None,
                   help='.npy of window starts, overriding --windows/--window-seed')
    p.add_argument('--batch', type=int, default=50, help='torch eval batch (default: 50)')
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()

    os.chdir(REPO)
    configs = [(d, *read_config(d)) for d in args.run_dirs]

    context_size = int(flag_value(configs[0][1], '--context-size', 32))
    val_files = {flag_value(argv, '--validation-data') for _, argv, _ in configs}
    if len(val_files) != 1:
        raise SystemExit(f'runs use different validation files: {sorted(val_files)}')

    starts = np.load(args.starts_file) if args.starts_file else None

    results = []
    for run_dir, argv, kind in configs:
        ckpt = checkpoint_path(run_dir, kind)
        if starts is None:
            # Token count is only known after tokenizing, so derive the starts once, from
            # the first run's validation stream, then reuse them for every run.
            from word_tokenizer import WordTokenizer
            enc = WordTokenizer(flag_value(argv, '--vocab-file'))
            with open(flag_value(argv, '--validation-data'), encoding='utf-8') as f:
                n_tokens = len(enc.encode(f.read()))
            rng = np.random.default_rng(args.window_seed)
            starts = rng.integers(0, n_tokens - context_size - 1, size=args.windows)
            print(f'Shared validation windows: {len(starts)} starts, seed '
                  f'{args.window_seed}, over {n_tokens} tokens')

        print(f'--- {run_dir}  [{kind}]')
        if kind == 'main':
            preds, targets, (loss_sum, count) = eval_main_checkpoint(
                argv, ckpt, starts)
        else:
            preds, targets, (loss_sum, count) = eval_torch_checkpoint(
                argv, ckpt, starts, args.batch, args.threads)
        valid = targets != IGNORE_INDEX
        correct = int((preds == targets)[valid].sum())
        results.append((run_dir, kind, loss_sum / count, correct / count, count,
                        preds, targets))

    print()
    print(f'{"run":60s} {"impl":6s} {"tokens":>8s} {"loss":>9s} {"acc":>9s}')
    for run_dir, kind, loss, acc, count, _, _ in results:
        print(f'{run_dir[-60:]:60s} {kind:6s} {count:8d} {loss:9.6f} {acc:9.6f}')

    if len(results) == 2:
        (_, _, _, _, _, pa, ta), (_, _, _, _, _, pb, tb) = results
        valid = ta != IGNORE_INDEX
        a_ok, b_ok = (pa == ta)[valid], (pb == tb)[valid]
        print()
        print(f'prediction agreement : {float((pa[valid] == pb[valid]).mean()):.4f}')
        print(f'only first correct   : {int((a_ok & ~b_ok).sum())}')
        print(f'only second correct  : {int((b_ok & ~a_ok).sum())}')


if __name__ == '__main__':
    main()
