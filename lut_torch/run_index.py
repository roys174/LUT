"""Discovery of training runs from ``results/<sweep>/<run>/config.txt``.

Deliberately stdlib-only -- no torch, no numpy. The point is that listing the available
models must not touch the checkpoints themselves: an LM checkpoint is up to 1.1 GB (the
baseline arm's output table alone is 944 MB), and ``torch.load``-ing a dozen of them just
to populate a dropdown would take minutes and tens of gigabytes of RAM.

Everything needed for the listing is already on the cheap side of that line. The run
script writes ``config.txt`` with the resolved paths and the verbatim training command,
so the task (POS vs plain LM), the arm, the seed and the checkpoint location are all
plain text; ``loss.csv`` carries the validation curve. The checkpoint is opened only once
a model is actually selected.

The task test is the *command*, not the directory name: a run is a language model when it
invoked ``lut_torch.train`` without ``--POS-task``, or ``gpt_baseline_train.py`` at all
(that trainer has no POS mode). Naming conventions drift between sweeps -- and
``results/GPT_POS`` is a third thing again, nanoGPT's own trainer, which is neither --
while the flag that decided ``vocab_size_output`` does not.

Two trainers appear here because their runs are directly comparable: same corpus, same
tokenizer, same context size, and the same validation windows at the same seed.
``engine`` records which one, so a client can say so without re-parsing the command.

What is *not* comparable across runs is a raw perplexity, once BPE runs
(``--tokenizer gpt2``, the WikiText-103 sweeps) sit in the same list as word-level ones.
``RunInfo.headline_metric`` picks the column that is; see the note on ``METRIC_PER_WORD``.
"""

import csv
import os
import shlex
from dataclasses import asdict, dataclass, field
from typing import List, Optional

TASK_LM = 'LM'
TASK_POS = 'POS'
TASK_OTHER = 'other'

# Which network a run trained. Both engines write the same `config.txt` and the same
# `loss.csv` columns, draw the same validation windows at the same seed, and take the
# same `--tokenizer` -- so for everything this module does they differ only by name, and
# a chat client can offer them side by side. They part company at the checkpoint, which
# is why `engine` is carried on RunInfo rather than re-derived later.
ENGINE_LUT = 'lut'
ENGINE_NANOGPT = 'nanogpt'

# Which column of `loss.csv` a run should be judged by.
#
# A perplexity is per *token*, and "token" is not one unit across these runs: the word
# tokenizer emits one per whitespace word, GPT-2 BPE emits ~1.16 per word on this corpus.
# Dividing the same total NLL by a larger count gives a smaller number, so ranking a BPE
# run against a word-level one by `perplexity` hands the BPE run a discount it did not
# earn -- the comparison measures tokenization, not modelling. The `_per_word` columns
# exist precisely to put both on one scale: same numerator (NLL over the same text), same
# denominator (whitespace words). They are written only where they are meaningful, which
# is why the choice is per run and not a global setting.
METRIC_PER_WORD = 'per_word'
METRIC_PER_TOKEN = 'per_token'

# The trainers, as they appear in the recorded command line.
TRAIN_MODULE = 'lut_torch.train'
GPT_TRAIN_SCRIPT = 'gpt_baseline_train.py'


@dataclass
class CheckpointInfo:
    """One ``.pt`` file belonging to a run."""

    kind: str                 # 'best' or 'last'
    path: str
    size_bytes: int
    mtime: float

    @property
    def label(self):
        return ('best (lowest val loss)' if self.kind == 'best'
                else 'last (most recent step)')


@dataclass
class RunMetrics:
    """A digest of ``loss.csv``: where the run is now and where it was best."""

    rows: int = 0
    last_step: Optional[int] = None
    last_epoch: Optional[float] = None
    best_step: Optional[int] = None
    best_epoch: Optional[float] = None
    best_loss: Optional[float] = None
    best_perplexity: Optional[float] = None
    best_accuracy: Optional[float] = None
    # The same headline numbers with <unk> targets removed. <unk> is 6.9% of this corpus
    # and its single most frequent token, so the two pairs differ enough to be worth
    # showing side by side rather than picking one. Word tokenizer only: under BPE id 0
    # is a real token and the trainer writes NaN here, which `_to_float` turns into None.
    best_loss_no_unk: Optional[float] = None
    best_perplexity_no_unk: Optional[float] = None
    best_accuracy_no_unk: Optional[float] = None
    # Total NLL per whitespace-delimited *word* rather than per token -- the last two
    # columns, added 2026-10-14. The mirror image of the pair above: written only for BPE
    # runs, NaN (-> None) for the word tokenizer, and None for every run that predates the
    # column. A nine-column file simply has no such key and `row.get` returns None.
    best_loss_per_word: Optional[float] = None
    best_perplexity_per_word: Optional[float] = None


@dataclass
class RunInfo:
    """One training run, as far as it can be known without opening a checkpoint."""

    run_id: str               # '<sweep>/<run directory>', stable and URL-safe enough
    sweep: str
    name: str
    task: str
    engine: str
    run_dir: str
    checkpoint_dir: str
    arm: str = ''
    seed: str = ''
    timestamp: str = ''
    vocab_file: str = ''
    tokenizer: str = ''
    training_data: str = ''
    context_size: Optional[int] = None
    git_commit: str = ''
    command: str = ''
    arm_flags: str = ''
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)

    @property
    def headline_metric(self):
        """``'per_word'`` for a BPE run, ``'per_token'`` for a word-level one.

        The tokenizer decides, not the presence of the column: a BPE run that has not yet
        written a per-word number is still a run whose per-token perplexity means
        something different from a word run's, and saying so is the whole point.
        """
        # `tokenizer` defaults to 'word' when the command recorded no --tokenizer, which
        # is what every pre-BPE run looks like. Anything else here is a sub-word scheme.
        return METRIC_PER_TOKEN if (self.tokenizer or 'word') == 'word' else METRIC_PER_WORD

    @property
    def headline_fallback(self):
        """True when :attr:`headline_loss` had to return a per-token number anyway.

        Only a BPE run can land here -- one still on its first validation, or one from
        before the columns existed. A client that labels the fallback "per word" is
        understating the perplexity by the tokens-per-word ratio (~1.16x on WikiText), so
        the flag is exposed rather than left for the caller to rediscover by comparing
        `headline_loss` against `best_loss`.
        """
        return (self.headline_metric == METRIC_PER_WORD
                and self.metrics.best_loss_per_word is None)

    @property
    def headline_loss(self):
        if (self.headline_metric == METRIC_PER_WORD
                and self.metrics.best_loss_per_word is not None):
            return self.metrics.best_loss_per_word
        return self.metrics.best_loss

    @property
    def headline_perplexity(self):
        if (self.headline_metric == METRIC_PER_WORD
                and self.metrics.best_perplexity_per_word is not None):
            return self.metrics.best_perplexity_per_word
        return self.metrics.best_perplexity

    def to_json(self):
        data = asdict(self)
        data['checkpoints'] = [dict(asdict(c), label=c.label) for c in self.checkpoints]
        # `asdict` walks fields, not properties, so the headline choice has to be written
        # in explicitly or the web UI would have to re-derive it from `tokenizer` -- and
        # the two copies of that rule would drift.
        data['headline_metric'] = self.headline_metric
        data['headline_loss'] = self.headline_loss
        data['headline_perplexity'] = self.headline_perplexity
        data['headline_fallback'] = self.headline_fallback
        return data

    def checkpoint(self, kind):
        for ckpt in self.checkpoints:
            if ckpt.kind == kind:
                return ckpt
        return None


# ---------------------------------------------------------------------------
# config.txt
# ---------------------------------------------------------------------------

def parse_config_txt(path):
    """``config.txt`` -> ``(fields, command)``.

    The file is ``key=value`` lines, then a blank line, then ``command:`` followed by the
    ``printf '%q '`` rendering of the argv the run actually used.
    """
    fields = {}
    command_lines = []
    in_command = False
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\n')
            if in_command:
                command_lines.append(line)
                continue
            if line.strip() == 'command:':
                in_command = True
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                if key and ' ' not in key:
                    fields[key] = value.strip()
    return fields, ' '.join(part for part in command_lines if part.strip()).strip()


def split_command(command):
    """``shlex.split`` that degrades to whitespace splitting on a malformed line."""
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def engine(tokens):
    """Which trainer produced the run, or ``''`` if it was neither of ours."""
    joined = ' '.join(tokens)
    if GPT_TRAIN_SCRIPT in joined:
        return ENGINE_NANOGPT
    if TRAIN_MODULE in joined:
        return ENGINE_LUT
    return ''


def classify(tokens):
    """The run's task, from the command line that produced it."""
    if not tokens:
        return TASK_OTHER
    which = engine(tokens)
    if not which:
        return TASK_OTHER
    # The GPT baseline exists only to be the next-token control for the LUT LM runs; it
    # has no POS mode, so there is no flag to consult.
    if which == ENGINE_NANOGPT:
        return TASK_LM
    # argparse's dest is POS_task and the flag is spelled --POS-task; accept either
    # casing so a hand-edited command still classifies.
    for token in tokens:
        if token.lower() in ('--pos-task', '--pos_task'):
            return TASK_POS
    return TASK_LM


def flag_value(tokens, *names):
    """The value following the first of ``names`` present in ``tokens``."""
    for index, token in enumerate(tokens):
        for name in names:
            if token == name and index + 1 < len(tokens):
                return tokens[index + 1]
            if token.startswith(name + '='):
                return token.split('=', 1)[1]
    return ''


def positional_argument(tokens):
    """The trainer's one positional argument: the training corpus.

    Scans from the trainer itself, which is ``-m lut_torch.train`` for one engine and a
    script path ending in ``gpt_baseline_train.py`` for the other -- never from argv[0],
    because that is an interpreter path that also looks positional.
    """
    start = None
    for index, token in enumerate(tokens):
        if token == TRAIN_MODULE or token.endswith(GPT_TRAIN_SCRIPT):
            start = index + 1
            break
    if start is None:
        return ''
    index = start
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith('-'):
            return token
        # Every option this trainer defines either is a store_true flag or takes exactly
        # one value, so skipping two on a value-looking successor is safe.
        if index + 1 < len(tokens) and not tokens[index + 1].startswith('-'):
            index += 2
        else:
            index += 1
    return ''


# ---------------------------------------------------------------------------
# loss.csv
# ---------------------------------------------------------------------------

def _to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number      # drop NaN (POS writes it)


def read_metrics(path):
    """Digest ``loss.csv``. Missing or unreadable -> an empty :class:`RunMetrics`."""
    metrics = RunMetrics()
    if not path or not os.path.exists(path):
        return metrics
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            best_row = None
            best_loss = None
            last_row = None
            count = 0
            for row in reader:
                row = {(k or '').strip(): (v or '').strip() for k, v in row.items()}
                # Still selected on per-token loss even for a BPE run: within one run the
                # validation windows are fixed, so tokens and words are the same counts in
                # every row and `loss_per_word` is `loss` times a constant > 0. The argmin
                # is therefore identical, and a run that wrote no per-word column still
                # gets a best row.
                loss = _to_float(row.get('loss'))
                last_row = row
                count += 1
                if loss is not None and (best_loss is None or loss < best_loss):
                    best_loss = loss
                    best_row = row
    except OSError:
        return metrics

    metrics.rows = count
    if last_row:
        step = _to_float(last_row.get('step'))
        metrics.last_step = int(step) if step is not None else None
        metrics.last_epoch = _to_float(last_row.get('epoch'))
    if best_row:
        step = _to_float(best_row.get('step'))
        metrics.best_step = int(step) if step is not None else None
        metrics.best_epoch = _to_float(best_row.get('epoch'))
        metrics.best_loss = _to_float(best_row.get('loss'))
        metrics.best_perplexity = _to_float(best_row.get('perplexity'))
        metrics.best_accuracy = _to_float(best_row.get('accuracy'))
        metrics.best_loss_no_unk = _to_float(best_row.get('loss_no_unk'))
        metrics.best_perplexity_no_unk = _to_float(best_row.get('perplexity_no_unk'))
        metrics.best_accuracy_no_unk = _to_float(best_row.get('accuracy_no_unk'))
        metrics.best_loss_per_word = _to_float(best_row.get('loss_per_word'))
        metrics.best_perplexity_per_word = _to_float(best_row.get('perplexity_per_word'))
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint locations
# ---------------------------------------------------------------------------

def checkpoint_paths(save_model, checkpoint_dir):
    """``(last, best)`` ``.pt`` paths, mirroring ``train.checkpoint_path``.

    ``--save-model X`` writes ``X.pt`` and ``X_best.pt``. The recorded flag is the
    authoritative source; ``checkpoint_dir`` is the fallback for a run whose command was
    not recorded.
    """
    if save_model:
        stem = save_model[:-3] if save_model.endswith('.pt') else save_model
        return stem + '.pt', stem + '_best.pt'
    if checkpoint_dir:
        return (os.path.join(checkpoint_dir, 'model.pt'),
                os.path.join(checkpoint_dir, 'model_best.pt'))
    return '', ''


def existing_checkpoints(save_model, checkpoint_dir):
    """The subset of those paths that exist, best first."""
    last_path, best_path = checkpoint_paths(save_model, checkpoint_dir)
    found = []
    for kind, path in (('best', best_path), ('last', last_path)):
        if not path or not os.path.exists(path):
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        # A checkpoint still being written has a real size but is not yet loadable; the
        # loader retries rather than the index guessing here.
        found.append(CheckpointInfo(kind=kind, path=path, size_bytes=stat.st_size,
                                    mtime=stat.st_mtime))
    return found


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def run_from_config(config_path, results_root, with_metrics=True):
    """Build a :class:`RunInfo` from one ``config.txt``, or ``None`` if unreadable.

    ``with_metrics=False`` skips ``loss.csv``, which is the expensive part: this tree
    holds a few hundred runs and only a handful are language models, so
    :func:`discover_runs` classifies first and reads the curve only for the survivors.
    """
    try:
        fields, command = parse_config_txt(config_path)
    except OSError:
        return None

    run_dir = os.path.dirname(config_path)
    rel = os.path.relpath(run_dir, results_root)
    sweep = rel.split(os.sep)[0] if os.sep in rel else ''
    name = os.path.basename(run_dir)

    tokens = split_command(command)
    task = classify(tokens)
    which = engine(tokens)
    checkpoint_dir = fields.get('checkpoint_dir', '')
    save_model = flag_value(tokens, '--save-model')
    context = flag_value(tokens, '--context-size')

    return RunInfo(
        run_id=rel.replace(os.sep, '/'),
        sweep=sweep or fields.get('sweep_name', ''),
        name=name,
        task=task,
        engine=which,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        arm=fields.get('arm', ''),
        seed=fields.get('seed', ''),
        timestamp=fields.get('run_timestamp', ''),
        vocab_file=flag_value(tokens, '--vocab-file'),
        tokenizer=flag_value(tokens, '--tokenizer') or 'word',
        training_data=positional_argument(tokens),
        context_size=int(context) if context.isdigit() else None,
        git_commit=fields.get('git_commit', ''),
        command=command,
        arm_flags=fields.get('arm_flags', ''),
        checkpoints=existing_checkpoints(save_model, checkpoint_dir),
        metrics=(read_metrics(os.path.join(run_dir, 'loss.csv')) if with_metrics
                 else RunMetrics()),
    )


def discover_runs(results_root, task=TASK_LM, require_checkpoint=True):
    """Every run under ``results_root`` matching ``task``, newest first.

    ``task=None`` returns all of them. ``require_checkpoint`` drops runs whose checkpoint
    directory is still empty -- a job that was queued and died leaves a full ``config.txt``
    and nothing to load.
    """
    runs = []
    if not os.path.isdir(results_root):
        return runs
    for sweep in sorted(os.listdir(results_root)):
        sweep_dir = os.path.join(results_root, sweep)
        if not os.path.isdir(sweep_dir):
            continue
        for entry in sorted(os.listdir(sweep_dir)):
            config_path = os.path.join(sweep_dir, entry, 'config.txt')
            if not os.path.isfile(config_path):
                continue
            run = run_from_config(config_path, results_root, with_metrics=False)
            if run is None:
                continue
            if task is not None and run.task != task:
                continue
            if require_checkpoint and not run.checkpoints:
                continue
            # Only now is the curve worth reading.
            run.metrics = read_metrics(os.path.join(run.run_dir, 'loss.csv'))
            runs.append(run)
    runs.sort(key=lambda r: (r.timestamp, r.name), reverse=True)
    return runs
