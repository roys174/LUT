"""Tests for the chat interface: run discovery, decoding, sampling, generation.

Split deliberately along the torch line, because the two interpreters available here
are too (see ``lut_torch/_testing.py``): the run-index tests are pure stdlib and run
anywhere, and everything that needs a model skips when torch is missing.

    .venv/bin/python test_torch_chat.py        # everything
    python3 test_torch_chat.py                 # the torch-free half

The load-bearing assertion is ``test_short_prompt_ignores_padding``: generation places a
short prompt at positions ``0..n-1`` and reads position ``n-1``, which is only correct
because attention is causal. That is a property of the model, not of this code, so it is
checked against the model rather than assumed.
"""

import os
import shutil
import tempfile

from lut_torch._testing import run_all, skip
from lut_torch import run_index
from lut_torch.run_index import (
    TASK_LM,
    TASK_OTHER,
    TASK_POS,
    checkpoint_paths,
    classify,
    discover_runs,
    flag_value,
    parse_config_txt,
    positional_argument,
    read_metrics,
    split_command,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


LM_COMMAND = (
    'python -m lut_torch.train nanoGPT_POS/data/wiki2/train_original_10_freq.txt '
    '--validation-data nanoGPT_POS/data/wiki2/val_original_10_freq.txt '
    '--vocab-file nanoGPT_POS/data/wiki2/vocab_file_10_freq.txt --tokenizer word '
    '--device cuda --context-size 32 --new-attention --output-code-dim 14 '
    '--save-model /ckpt/LM/run/model --output-code-bias'
)
POS_COMMAND = LM_COMMAND.replace('--tokenizer word', '--tokenizer word --POS-task')


def write_run(root, sweep, name, command, loss_rows=(), checkpoints=()):
    """Materialise a run directory the way the sbatch scripts do."""
    run_dir = os.path.join(root, sweep, name)
    os.makedirs(run_dir, exist_ok=True)
    save_model = flag_value(split_command(command), '--save-model')
    checkpoint_dir = os.path.dirname(save_model)
    with open(os.path.join(run_dir, 'config.txt'), 'w') as handle:
        handle.write(f"run_timestamp={name.split('_')[0]}\n")
        handle.write(f"run_dir={run_dir}\n")
        handle.write(f"checkpoint_dir={checkpoint_dir}\n")
        handle.write("arm=an arm with spaces\nseed=42\n")
        handle.write("git_commit=abc1234\n")
        handle.write("\ncommand:\n")
        handle.write(command + ' \n')
    if loss_rows:
        with open(os.path.join(run_dir, 'loss.csv'), 'w') as handle:
            handle.write('step, epoch, tokens, loss, perplexity, accuracy, '
                         'loss_no_unk, perplexity_no_unk, accuracy_no_unk\n')
            for row in loss_rows:
                handle.write(', '.join(str(v) for v in row) + '\n')
    for kind in checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        suffix = '_best.pt' if kind == 'best' else '.pt'
        with open(save_model + suffix, 'wb') as handle:
            handle.write(b'x' * 1024)
    return run_dir


# ---------------------------------------------------------------------------
# config.txt parsing
# ---------------------------------------------------------------------------

def test_parse_config_txt_splits_fields_from_command():
    root = tempfile.mkdtemp()
    try:
        run_dir = write_run(root, 'LM_sweep', '20260917-093002_arm_seed42', LM_COMMAND)
        fields, command = parse_config_txt(os.path.join(run_dir, 'config.txt'))
        assert fields['run_timestamp'] == '20260917-093002'
        assert fields['checkpoint_dir'] == '/ckpt/LM/run'
        # An `=` inside a value must not confuse the key split.
        assert fields['arm'] == 'an arm with spaces'
        assert command.startswith('python -m lut_torch.train ')
        assert '--output-code-bias' in command
    finally:
        shutil.rmtree(root)


def test_classify_reads_the_flag_not_the_name():
    assert classify(split_command(LM_COMMAND)) == TASK_LM
    assert classify(split_command(POS_COMMAND)) == TASK_POS
    # A run whose directory says LM but whose command is POS is a POS run.
    assert classify(split_command(POS_COMMAND.replace('wiki2', 'LM_wiki2'))) == TASK_POS
    # Anything that is not this trainer (results/GPT_POS, for one) is neither.
    assert classify(split_command('python nanoGPT/train.py config/foo.py')) == TASK_OTHER
    assert classify([]) == TASK_OTHER


def test_flag_and_positional_extraction():
    tokens = split_command(LM_COMMAND)
    assert flag_value(tokens, '--vocab-file').endswith('vocab_file_10_freq.txt')
    assert flag_value(tokens, '--context-size') == '32'
    assert flag_value(tokens, '--missing-flag') == ''
    # The corpus is the single positional, and it follows the module name -- not any of
    # the option values that also look like paths.
    assert positional_argument(tokens).endswith('train_original_10_freq.txt')


def test_checkpoint_paths_mirror_the_trainer():
    last, best = checkpoint_paths('/ckpt/run/model', '/ckpt/run')
    assert (last, best) == ('/ckpt/run/model.pt', '/ckpt/run/model_best.pt')
    # --save-model may already carry the extension.
    assert checkpoint_paths('/ckpt/run/model.pt', '')[1] == '/ckpt/run/model_best.pt'
    # With no recorded command, fall back to the directory's conventional names.
    assert checkpoint_paths('', '/ckpt/run') == ('/ckpt/run/model.pt',
                                                 '/ckpt/run/model_best.pt')


# ---------------------------------------------------------------------------
# loss.csv
# ---------------------------------------------------------------------------

def test_read_metrics_takes_the_minimum_loss_row():
    root = tempfile.mkdtemp()
    try:
        run_dir = write_run(root, 'LM_sweep', '20260917-1_a_seed42', LM_COMMAND, loss_rows=[
            (0, 0.0, 0, 9.57, 14398.0, 0.087, 9.57, 14397.9, 0.0),
            (211, 0.10, 216064, 5.10, 164.0, 0.23, 5.40, 221.4, 0.21),
            (422, 0.20, 432128, 5.30, 200.3, 0.24, 5.60, 270.4, 0.22),
        ])
        metrics = read_metrics(os.path.join(run_dir, 'loss.csv'))
        assert metrics.rows == 3
        # Best is the lowest loss, which is not the last row.
        assert metrics.best_step == 211
        assert abs(metrics.best_loss - 5.10) < 1e-9
        assert abs(metrics.best_perplexity - 164.0) < 1e-9
        assert abs(metrics.best_accuracy_no_unk - 0.21) < 1e-9
        # The last row is reported separately, so "best" and "where it is now" both show.
        assert metrics.last_step == 422
    finally:
        shutil.rmtree(root)


def test_read_metrics_drops_nan_and_tolerates_missing_file():
    root = tempfile.mkdtemp()
    try:
        # POS runs write nan into the three no-unk columns; those must not become
        # "0.0 perplexity" in the UI.
        run_dir = write_run(root, 'POS_sweep', '20260917-1_a_seed42', POS_COMMAND,
                            loss_rows=[(10, 0.1, 100, 0.5, 1.6, 0.94,
                                        'nan', 'nan', 'nan')])
        metrics = read_metrics(os.path.join(run_dir, 'loss.csv'))
        assert metrics.best_loss == 0.5
        assert metrics.best_loss_no_unk is None
        assert read_metrics(os.path.join(run_dir, 'absent.csv')).rows == 0
        assert read_metrics(None).best_loss is None
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_discover_runs_selects_lm_with_checkpoints():
    root = tempfile.mkdtemp()
    try:
        write_run(root, 'LM_sweep', '20260917-093002_learnedD14_seed42',
                  LM_COMMAND.replace('/ckpt/LM/run', '/'.join([root, 'ck', 'a']))
                  , loss_rows=[(1, 0.1, 10, 5.1, 164.0, 0.23, 5.4, 221.4, 0.21)],
                  checkpoints=('best', 'last'))
        # Same sweep, but the job died before writing anything.
        write_run(root, 'LM_sweep', '20260916-195144_learnedD64_seed42',
                  LM_COMMAND.replace('/ckpt/LM/run', '/'.join([root, 'ck', 'b'])))
        # A POS run and a non-trainer run, both of which must stay out of the list.
        write_run(root, 'POS_sweep', '20260917-1_baseline_seed42',
                  POS_COMMAND.replace('/ckpt/LM/run', '/'.join([root, 'ck', 'c'])),
                  checkpoints=('best',))
        write_run(root, 'GPT_POS', '20260917-1_gpt',
                  'python nanoGPT/train.py --save-model ' + root + '/ck/d/model',
                  checkpoints=('best',))

        runs = discover_runs(root, task=TASK_LM)
        assert [r.name for r in runs] == ['20260917-093002_learnedD14_seed42']
        run = runs[0]
        assert run.run_id == 'LM_sweep/20260917-093002_learnedD14_seed42'
        assert run.sweep == 'LM_sweep'
        assert run.context_size == 32
        assert run.tokenizer == 'word'
        assert run.vocab_file.endswith('vocab_file_10_freq.txt')
        assert sorted(c.kind for c in run.checkpoints) == ['best', 'last']
        assert run.checkpoint('best').path.endswith('_best.pt')
        assert abs(run.metrics.best_loss - 5.1) < 1e-9

        # Without the checkpoint filter the dead run reappears, newest first.
        both = discover_runs(root, task=TASK_LM, require_checkpoint=False)
        assert len(both) == 2
        assert both[0].timestamp > both[1].timestamp
        assert len(discover_runs(root, task=TASK_POS)) == 1
        assert len(discover_runs(root, task=None, require_checkpoint=False)) == 4
    finally:
        shutil.rmtree(root)


def test_discover_runs_survives_a_missing_root():
    assert discover_runs('/nonexistent/results') == []


def test_run_json_is_serialisable():
    import json

    root = tempfile.mkdtemp()
    try:
        write_run(root, 'LM_sweep', '20260917-1_a_seed42',
                  LM_COMMAND.replace('/ckpt/LM/run', root + '/ck/a'),
                  checkpoints=('best',))
        run = discover_runs(root, task=TASK_LM)[0]
        payload = json.loads(json.dumps(run.to_json()))
        assert payload['checkpoints'][0]['label'].startswith('best')
        assert payload['metrics']['best_loss'] is None
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# Decoding and sampling (torch)
# ---------------------------------------------------------------------------

def _require_torch():
    if not HAS_TORCH:
        skip('torch is not importable on this interpreter')


class FakeTokenizer:
    """The slice of ``WordTokenizer`` the decoder uses."""

    def __init__(self, tokens):
        self.id_to_token = list(tokens)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    @property
    def n_vocab(self):
        return len(self.id_to_token)

    def tokens_from_ids(self, ids):
        return [self.id_to_token[int(i)] for i in ids]

    def decode(self, ids):
        # Copied verbatim from word_tokenizer.WordTokenizer.decode: the point of the
        # incremental decoder is to match this, so it must not be paraphrased here.
        parts = self.tokens_from_ids(ids)
        out = []
        for i, tok in enumerate(parts):
            if tok == '<eol>':
                out.append('\n')
            elif i == 0 or out[-1] == '\n':
                out.append(tok)
            elif len(tok) == 1 and not tok.isalnum():
                out.append(tok)
            else:
                out.append(' ' + tok)
        return ''.join(out)


FAKE_VOCAB = ['<unk>', '<eol>', 'the', 'cat', 'sat', '.', ',', 'on', 'mat']


def test_incremental_decode_matches_whole():
    _require_torch()
    from lut_torch.inference import IncrementalDecoder

    enc = FakeTokenizer(FAKE_VOCAB)
    sequences = [
        [2, 3, 4, 5],                    # the cat sat .
        [1, 2, 3, 1, 5],                 # newlines at both ends of a phrase
        [5, 6, 2],                       # opening on punctuation
        [0, 2, 8],                       # <unk> is a word like any other
    ]
    for ids in sequences:
        decoder = IncrementalDecoder(enc)
        streamed = ''.join(decoder.feed(i) for i in ids)
        assert streamed == enc.decode(ids), (ids, streamed, enc.decode(ids))


def test_incremental_decode_continues_a_prompt():
    _require_torch()
    from lut_torch.inference import IncrementalDecoder

    enc = FakeTokenizer(FAKE_VOCAB)
    prompt, reply = [2, 3], [4, 5]
    decoder = IncrementalDecoder.continuing(enc, prompt)
    streamed = ''.join(decoder.feed(i) for i in reply)
    # A continuation must space its first token against the prompt, not against nothing.
    assert enc.decode(prompt) + streamed == enc.decode(prompt + reply)

    # A prompt ending in <eol> leaves the next token at the start of a line.
    decoder = IncrementalDecoder.continuing(enc, [2, 1])
    assert decoder.feed(3) == 'cat'


def test_encode_prompt_keeps_a_trailing_eol_only_when_asked():
    """The encoder ends every non-empty line with ``<eol>``; a prompt is not a line."""
    _require_torch()
    from lut_torch.inference import encode_prompt

    class Encoder(FakeTokenizer):
        def encode(self, text):
            # word_tokenizer._split, reduced to the whitespace case these tests need.
            ids = []
            lines = text.split('\n')
            for i, line in enumerate(lines):
                ids.extend(self.token_to_id.get(w, 0) for w in line.split())
                if i < len(lines) - 1 or line:
                    ids.append(1)
            return ids

    enc = Encoder(FAKE_VOCAB)
    # Without the trim the model is told the sentence ended and a new line begins, which
    # in this corpus means a heading.
    assert enc.encode('the cat')[-1] == 1
    assert encode_prompt(enc, 'the cat') == [2, 3]
    # A user who did end with a newline meant it.
    assert encode_prompt(enc, 'the cat\n') == [2, 3, 1]
    # Line breaks inside the prompt are untouched; only the last one is in question.
    assert encode_prompt(enc, 'the cat\nsat') == [2, 3, 1, 4]
    assert encode_prompt(enc, '') == []


def test_filter_logits_bans_and_restricts():
    _require_torch()
    import torch as _torch
    from lut_torch.inference import filter_logits

    logits = _torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    banned = filter_logits(logits, banned_ids=(0,))
    assert banned[0] == float('-inf')
    assert banned[1] == 4.0

    top_k = filter_logits(logits, top_k=2)
    assert _torch.isfinite(top_k).sum().item() == 2

    # A ban must be applied before top-k, or it eats one of the k slots.
    both = filter_logits(logits, top_k=2, banned_ids=(0,))
    assert _torch.isfinite(both).sum().item() == 2
    assert both[0] == float('-inf') and _torch.isfinite(both[1])

    # Temperature 0.5 doubles every logit gap.
    hot = filter_logits(logits, temperature=0.5)
    assert abs(float(hot[0] - hot[1]) - 2.0) < 1e-6


def test_top_p_keeps_a_nonempty_nucleus():
    _require_torch()
    import torch as _torch
    from lut_torch.inference import filter_logits

    # One token holds 99.9% of the mass, which is already past any top_p below it.
    peaked = _torch.tensor([20.0, 0.0, 0.0, 0.0])
    kept = filter_logits(peaked, top_p=0.5)
    assert _torch.isfinite(kept).sum().item() == 1

    flat = _torch.zeros(10)
    kept = filter_logits(flat, top_p=0.45)
    # Uniform over 10: the smallest prefix reaching 0.45 is 5 tokens.
    assert _torch.isfinite(kept).sum().item() == 5


# ---------------------------------------------------------------------------
# The window (torch, builds a small model)
# ---------------------------------------------------------------------------

def _tiny_model():
    """A small LutModel in the shape the LM sweeps use, wrapped for chat."""
    import numpy as np
    import torch as _torch
    from lut_torch.config import LutConfig
    from lut_torch.inference import LutChatModel
    from lut_torch.modules import LutModel

    np.random.seed(0)
    _torch.manual_seed(0)
    cfg = LutConfig(context_size=8, embedding_dim=8, positional_dim=3, num_layers=2,
                    num_heads=2, vocab_size_input=len(FAKE_VOCAB),
                    vocab_size_output=len(FAKE_VOCAB), n_t=4, n_c=3, attn_qk_dim=3,
                    attn_qk_n_c=3, attn_qk_n_t=4, new_MHA=True, qkpe_direct_sign=True,
                    freeze_PE=True, stack_heads=True, dropout=0.2)
    model = LutModel(cfg)
    # Random tables, so the logits actually depend on the input.
    with _torch.no_grad():
        for table in model.table_parameters():
            table.normal_(0.0, 0.5)
    model.eval()
    enc = FakeTokenizer(FAKE_VOCAB)
    return LutChatModel(model, cfg, {'tokenizer': 'word'}, 0, enc,
                        _torch.device('cpu'), 'memory')


def test_short_prompt_ignores_padding():
    """The whole short-prompt scheme rests on attention being causal."""
    _require_torch()
    import torch as _torch

    chat = _tiny_model()
    prompt = [2, 3, 4]
    base = chat.next_logits(prompt)
    for filler in (0, 5, 8):
        tokens = _torch.tensor([prompt + [filler] * (chat.context_size - len(prompt))],
                               dtype=_torch.int64)
        with _torch.no_grad():
            logits = chat.model(tokens)[0, len(prompt) - 1].float()
        assert _torch.allclose(base, logits, atol=0, rtol=0), (
            'a token after the prompt changed the prediction at the prompt\'s last '
            'position -- attention is not causal, and the padding scheme is invalid')


def test_window_slides_once_the_prompt_is_long():
    _require_torch()
    import torch as _torch

    chat = _tiny_model()
    long_prompt = [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 7, 8]     # 12 > context_size 8
    tail = long_prompt[-chat.context_size:]
    assert _torch.allclose(chat.next_logits(long_prompt), chat.next_logits(tail))
    # And the tail alone is a full window, so it is read at the last position.
    tokens, read_at = chat._window(long_prompt)
    assert read_at == chat.context_size - 1
    assert tokens.shape == (1, chat.context_size)
    assert tokens[0].tolist() == tail


def test_eval_mode_makes_generation_deterministic():
    _require_torch()
    import torch as _torch

    chat = _tiny_model()
    assert not chat.model.training, 'dropout would be live'
    prompt = [2, 3]
    first = chat.next_logits(prompt)
    second = chat.next_logits(prompt)
    assert _torch.allclose(first, second)


def test_generate_is_reproducible_and_respects_its_limits():
    _require_torch()
    chat = _tiny_model()
    prompt = [2, 3]

    def sample(**kwargs):
        return list(chat.generate(prompt, max_new_tokens=6, seed=7, **kwargs))

    first, second = sample(), sample()
    assert [e['id'] for e in first] == [e['id'] for e in second]
    assert len(first) == 6
    assert all(set(event) >= {'id', 'token', 'text', 'prob'} for event in first)

    # suppress_unk must hold for every step, not just the first.
    for event in sample(suppress_unk=True, temperature=1.5, top_k=0, top_p=1.0):
        assert event['id'] != 0

    # Greedy is greedy: temperature 0 reproduces argmax without a seed.
    greedy = [e['id'] for e in chat.generate(prompt, max_new_tokens=4, temperature=0.0)]
    again = [e['id'] for e in chat.generate(prompt, max_new_tokens=4, temperature=0.0)]
    assert greedy == again


def test_generate_stops_when_asked():
    _require_torch()
    chat = _tiny_model()
    produced = []

    # should_stop is how the server abandons a generation whose browser has gone.
    def should_stop():
        return len(produced) >= 3

    for event in chat.generate([2, 3], max_new_tokens=50, seed=1,
                               should_stop=should_stop):
        produced.append(event)
    assert len(produced) == 3

    # stop_at_eol ends the reply at the first newline token.
    ids = [e['id'] for e in chat.generate([2, 3], max_new_tokens=50, seed=3,
                                          stop_at_eol=True, suppress_unk=False)]
    assert 1 not in ids[:-1]


def test_config_round_trips_through_a_checkpoint_dict():
    """``to_dict`` carries derived fields that the constructor refuses."""
    _require_torch()
    from lut_torch.config import LutConfig
    from lut_torch.inference import config_from_checkpoint

    original = LutConfig(context_size=16, num_heads=2, new_MHA=True, weight_sharing=8,
                         n_t=16, output_code_dim=5, output_code_bias=True)
    restored = config_from_checkpoint({'config': original.to_dict()})
    assert restored.to_dict() == original.to_dict()

    try:
        config_from_checkpoint({'config': dict(original.to_dict(), from_the_future=1)})
    except ValueError as exc:
        assert 'from_the_future' in str(exc)
    else:
        raise AssertionError('a config field this tree does not know must be reported')


# ---------------------------------------------------------------------------
# Server routing (no torch: the handler is exercised through the service object)
# ---------------------------------------------------------------------------

def test_service_module_imports_without_torch():
    """``--help`` and a bad results root must not pay for importing torch."""
    import importlib

    module = importlib.import_module('lut_torch.chat_server')
    parser = module.build_parser()
    args = parser.parse_args(['--port', '9999'])
    assert args.port == 9999
    assert args.task == run_index.TASK_LM
    assert os.path.isdir(module.STATIC_DIR)
    for name in ('index.html', 'app.js', 'style.css'):
        assert os.path.isfile(os.path.join(module.STATIC_DIR, name)), name


class _StubRun:
    run_id = 'LM_sweep/run_a'
    sweep = 'LM_sweep'
    name = 'run_a'

    def to_json(self):
        return {'run_id': self.run_id, 'sweep': self.sweep, 'name': self.name,
                'arm': 'learnedD14', 'metrics': {'best_loss': 5.1},
                'checkpoints': [{'kind': 'best', 'label': 'best', 'size_bytes': 1}]}


class _StubCheckpoint:
    kind = 'best'
    path = '/ckpt/model_best.pt'


class _StubModel:
    context_size = 4

    def encode(self, text):
        return [2, 3, 4] if text.strip() else []

    def token_strings(self, ids):
        return ['tok%d' % i for i in ids]

    def describe(self):
        return {'context_size': 4, 'step': 10, 'num_layers': 2, 'num_heads': 2,
                'vocab_size': 9, 'table_entries': 100, 'output_code': None,
                'device': 'cpu', 'checkpoint': _StubCheckpoint.path,
                'tokenizer': 'word', 'vocab_file': 'v.txt'}

    def top_next(self, ids, k=8, exclude_unk=False):
        return [{'id': 2, 'token': 'the', 'prob': 0.5}]

    def generate(self, prompt_ids, should_stop=None, **kwargs):
        for _ in range(kwargs.get('max_new_tokens', 3)):
            if should_stop is not None and should_stop():
                return
            yield {'id': 2, 'token': 'the', 'text': ' the', 'prob': 0.5}


class _StubService:
    """Enough of :class:`ChatService` to exercise routing without loading a checkpoint."""

    device = 'cpu'
    results_root = '/results'

    def __init__(self):
        import threading

        self.generation_lock = threading.Lock()
        self.cache = type('C', (), {'loaded': staticmethod(list)})()

    def refresh(self):
        return [_StubRun()]

    def tokenize(self, run_id, text):
        if run_id != _StubRun.run_id:
            raise KeyError(f"unknown run {run_id!r}")
        return {'ids': [2, 0], 'tokens': ['the', '<unk>'], 'unk': [False, True],
                'count': 2, 'unk_count': 1, 'context_size': 4, 'dropped': 0}

    def model(self, run_id, kind='best'):
        if run_id != _StubRun.run_id:
            raise KeyError(f"unknown run {run_id!r}")
        return _StubRun(), _StubCheckpoint(), _StubModel()


class _Server:
    """A live handler on a throwaway port, with the stub service behind it."""

    def __init__(self):
        import threading
        from http.server import ThreadingHTTPServer
        from lut_torch.chat_server import ChatHandler

        self.previous = ChatHandler.service
        ChatHandler.service = _StubService()
        self.handler = ChatHandler
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), ChatHandler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = 'http://127.0.0.1:%d' % self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.handler.service = self.previous

    def get(self, path):
        import urllib.request

        with urllib.request.urlopen(self.base + path) as response:
            return response.status, response.headers.get('Content-Type'), response.read()

    def post(self, path, payload):
        import json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()


def test_static_routes_serve_the_ui_and_refuse_traversal():
    import urllib.error

    server = _Server()
    try:
        status, content_type, body = server.get('/')
        assert status == 200 and b'<title>LUT chat</title>' in body
        assert 'text/html' in content_type
        for path, marker in (('/static/app.js', b'const state'),
                             ('/static/style.css', b'--accent')):
            status, _, body = server.get(path)
            assert status == 200 and marker in body, path
        try:
            server.get('/static/../../lut_torch/train.py')
        except urllib.error.HTTPError as exc:
            assert exc.code in (403, 404)
        else:
            raise AssertionError('the static route escaped chat_ui/')
    finally:
        server.close()


def test_api_contract_matches_what_the_page_reads():
    """Every field ``app.js`` touches, checked against the routes that produce it."""
    import json

    server = _Server()
    try:
        status, _, body = server.get('/api/runs')
        listing = json.loads(body)
        assert status == 200
        assert listing['runs'][0]['run_id'] == _StubRun.run_id
        assert listing['device'] == 'cpu' and listing['results_root'] == '/results'

        status, body = server.post('/api/tokenize',
                                   {'run_id': _StubRun.run_id, 'text': 'the qqq'})
        tokens = json.loads(body)
        assert status == 200
        assert set(tokens) >= {'ids', 'tokens', 'unk', 'count', 'unk_count',
                               'context_size', 'dropped'}

        status, body = server.post('/api/model',
                                   {'run_id': _StubRun.run_id, 'kind': 'best'})
        described = json.loads(body)['model']
        assert status == 200
        assert set(described) >= {'num_layers', 'num_heads', 'context_size',
                                  'vocab_size', 'table_entries', 'output_code',
                                  'step', 'device'}

        # An unknown run is a 404 with a message, not a traceback.
        status, body = server.post('/api/tokenize', {'run_id': 'nope', 'text': 'x'})
        assert status == 404 and 'error' in json.loads(body)
        assert server.post('/api/nope', {})[0] == 404
    finally:
        server.close()


def test_generate_streams_ndjson():
    import json

    server = _Server()
    try:
        status, body = server.post('/api/generate', {
            'run_id': _StubRun.run_id, 'prompt': 'the cat sat', 'max_new_tokens': 3})
        assert status == 200
        events = [json.loads(line) for line in body.decode().splitlines() if line.strip()]
        assert [e['type'] for e in events] == ['start'] + ['token'] * 3 + ['done']
        assert events[0]['context'] and events[0]['top_next']
        assert all(e['text'] == ' the' for e in events[1:4])
        assert events[-1]['tokens'] == 3

        # A prompt that tokenizes to nothing is the caller's error, not a 500.
        assert server.post('/api/generate',
                           {'run_id': _StubRun.run_id, 'prompt': '   '})[0] == 400
    finally:
        server.close()


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
