"""Pre-tokenized .bin input, BPE word counting, per-word loss and the no-unk fix.

    .venv313/bin/python test_token_bin.py        # shim runner (no pytest needed)
"""

import gc
import hashlib
import json
import math
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpt_baseline_train as gbt
import lut_torch.train as lt
from lut_torch._testing import run_all
from lut_torch.data import (IGNORE_INDEX, SnippetDataset, WordTables, load_datasets,
                            read_token_bin_manifest)

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, 'nanoGPT_POS', 'data', 'wiki2')
VAL = os.path.join(DATA, 'val_original_10_freq.txt')
VOCAB = os.path.join(DATA, 'vocab_file_10_freq.txt')

ENC = tiktoken.get_encoding('gpt2')
TABLES = WordTables(ENC)


def _write_bin(directory, name, text, tokenizer='gpt2', manifest_edit=None):
    tokens = np.array(ENC.encode(text), dtype=np.uint16)
    path = os.path.join(directory, name)
    tokens.tofile(path)
    manifest = {'format': 'lut-token-bin-v1', 'tokenizer': tokenizer, 'dtype': 'uint16',
                'n_tokens': int(len(tokens)), 'n_words': len(text.split()),
                'tokens_sha256': hashlib.sha256(tokens.tobytes()).hexdigest()}
    if manifest_edit:
        manifest.update(manifest_edit)
    with open(path + '.json', 'w') as f:
        json.dump(manifest, f)
    return path, tokens


def _starts(ids):
    ids = np.asarray(ids)
    return TABLES.starts_word(np.concatenate([[-1], ids[:-1]]), ids)


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def test_word_starts_equal_split_count_on_real_text():
    text = _read(VAL)
    ids = ENC.encode(text)
    assert int(_starts(ids).sum()) == len(text.split())


def test_word_start_hand_cases():
    def n(text):
        return int(_starts(ENC.encode(text)).sum())
    assert n('a walking') == 2
    assert n(' walk' + 'ing') == 1            # "ing" continues " walk"
    assert n('\n') == 0 and n('\n\n') == 0
    assert n('x\n\n the') == 2
    assert n('walking') == 1
    ids = ENC.encode('a walking')
    flags = _starts(ids).tolist()
    assert flags[0] and flags[1]


class _RandomLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 16)
        self.out = nn.Linear(16, vocab)

    def forward(self, x):
        return self.out(self.emb(x))


def test_evaluate_per_word_matches_independent_computation():
    text = 'The quick brown fox\n\njumps over the lazy dog, walking and talking.\n' * 6
    ids = np.array(ENC.encode(text), dtype=np.int32)
    ds = SnippetDataset(ids, None, 8, False, unk_id=None, word_tables=TABLES)
    torch.manual_seed(0)
    model = _RandomLM(ENC.n_vocab)
    starts = np.array([0, 3, 10, 20])
    args = SimpleNamespace(validation_batch_size=3, batch_size=3,
                           lm_validate_last_position_only=False)
    cfg = SimpleNamespace(POS_task=False, vocab_size_output=ENC.n_vocab)
    val = lt.evaluate(model, ds, starts, args, cfg, torch.device('cpu'))

    nll, words = 0.0, 0
    for s in starts:
        window = ids[s:s + 9].astype(np.int64)
        with torch.no_grad():
            logits = model(torch.from_numpy(window[:-1])[None])[0]
        nll += float(F.cross_entropy(logits, torch.from_numpy(window[1:]), reduction='sum'))
        for i in range(1, 9):
            piece = ENC.decode_single_token_bytes(int(window[i]))
            prev = ENC.decode_single_token_bytes(int(window[i - 1]))
            ws = b' \n\t\r\x0b\x0c'
            if (any(c not in ws for c in piece)
                    and (piece[0] in ws or prev[-1] in ws)):
                words += 1
    assert val['words'] == words > 0
    assert math.isclose(val['loss_per_word'], nll / words, rel_tol=1e-5)
    assert val['count'] == 8 * len(starts)


def test_no_unk_nan_for_gpt2_and_unchanged_for_word():
    ids = np.arange(200, dtype=np.int32) % 7        # lots of id 0
    cfg = SimpleNamespace(POS_task=False, vocab_size_output=7)
    args = SimpleNamespace(validation_batch_size=4, batch_size=4,
                           lm_validate_last_position_only=False)
    starts = np.arange(0, 50, 5)
    torch.manual_seed(0)
    model = _RandomLM(7)
    bpe = SnippetDataset(ids, None, 8, False, unk_id=None, word_tables=None)
    v = lt.evaluate(model, bpe, starts, args, cfg, torch.device('cpu'))
    assert math.isnan(v['loss_no_unk']) and math.isnan(v['acc_no_unk'])
    assert v['count_no_unk'] == 0 and math.isnan(v['loss_per_word']) and v['words'] == 0
    word = SnippetDataset(ids, None, 8, False)       # default: id 0 is <unk>
    v = lt.evaluate(model, word, starts, args, cfg, torch.device('cpu'))
    assert 0 < v['count_no_unk'] < v['count']
    assert v['loss_no_unk'] == v['loss_no_unk'] and math.isnan(v['loss_per_word'])


def _args(train, val, tokenizer='gpt2', pos=False):
    return SimpleNamespace(training_data=train, validation_data=val, context_size=8,
                           POS_task=pos, tokenizer=tokenizer, vocab_file=None)


def test_bin_round_trip_matches_text_path():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        text = _read(VAL)[:20000]
        txt = os.path.join(tmp, 'a.txt')
        with open(txt, 'w', encoding='utf-8') as f:
            f.write(text)
        binp, tokens = _write_bin(tmp, 'a.bin', text)
        t_text, v_text = load_datasets(_args(txt, txt), ENC)
        t_bin, v_bin = load_datasets(_args(binp, binp), ENC)
        assert t_bin.tokens.dtype == np.uint16 and t_text.tokens.dtype == np.int32
        np.testing.assert_array_equal(t_text.tokens, t_bin.tokens)
        assert t_text.length == v_bin.length
        starts = np.array([0, 5, 100, t_text.length - 1])
        a, b = t_text.batch(starts), v_bin.batch(starts)
        for x, y in zip(a, b):
            assert x.dtype == y.dtype == np.int64
            np.testing.assert_array_equal(x, y)
        assert v_bin.unk_id is None and v_bin.word_tables is not None
        # Release the memmaps so the (NFS) temp dir can be removed.
        del t_bin, v_bin, a, b
        gc.collect()


def test_bad_manifests_and_mixed_inputs_raise():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        text = 'hello world, this is some text for tokenizing in a test. ' * 5
        good, _ = _write_bin(tmp, 'g.bin', text)
        bad_size, _ = _write_bin(tmp, 's.bin', text, manifest_edit={'n_tokens': 7})
        bad_tok, _ = _write_bin(tmp, 't.bin', text, tokenizer='r50k_base')
        bad_fmt, _ = _write_bin(tmp, 'f.bin', text, manifest_edit={'format': 'nope'})
        txt = os.path.join(tmp, 'a.txt')
        with open(txt, 'w') as f:
            f.write(text)

        def raises(fn, needle):
            try:
                fn()
            except ValueError as e:
                assert needle in str(e), str(e)
                return
            raise AssertionError('expected ValueError')
        raises(lambda: read_token_bin_manifest(bad_size), 'bytes')
        raises(lambda: read_token_bin_manifest(bad_fmt), 'format')
        raises(lambda: load_datasets(_args(bad_tok, bad_tok), ENC), 'r50k_base')
        raises(lambda: load_datasets(_args(good, txt), ENC), 'both')
        raises(lambda: load_datasets(_args(good, good, pos=True), ENC), 'POS')
        raises(lambda: load_datasets(_args(good, good, tokenizer='word'), ENC), 'gpt2')


def test_gpt_baseline_end_to_end_gpt2_bin():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        binp, _ = _write_bin(tmp, 'v.bin', _read(VAL)[:60000])
        loss_file = os.path.join(tmp, 'loss.csv')
        gbt.main([binp, '--validation-data', binp, '--tokenizer', 'gpt2',
                  '--n-layer', '1', '--n-head', '2', '--n-embd', '16',
                  '--batch-size', '4', '--max-steps', '5', '--warmup-steps', '2',
                  '--validation-interval', '2', '--testing-length', '20',
                  '--validation-batch-size', '7', '--device', 'cpu',
                  '--loss-file', loss_file])
        with open(loss_file) as f:
            lines = f.read().strip().splitlines()
        assert len(lines[0].split(',')) == 11
        rows = [[float(x) for x in line.split(',')] for line in lines[1:]]
        assert len(rows) == 3
        for r in rows:
            assert len(r) == 11
            assert math.isnan(r[6]) and math.isnan(r[8])       # no-unk under BPE
            assert math.isfinite(r[10]) and r[10] > r[4]       # tokens/word > 1
        assert abs(rows[0][3] - math.log(50257)) < 1.0


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
