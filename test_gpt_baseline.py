"""Checks that gpt_baseline_train.py differs from a lut_torch LM run only in the model.

    .venv/bin/python test_gpt_baseline.py        # shim runner (venv has no pytest)
    python -m pytest -q test_gpt_baseline.py      # if pytest and torch are both present

Uses the real wiki2 validation file as both train and validation data so the windows
are drawn over the real stream length; everything runs on CPU in a few seconds.
"""

import math
import os
import tempfile

import numpy as np
import torch

import gpt_baseline_train as gbt
import lut_torch.train as lt
from lut_torch._testing import run_all

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, 'nanoGPT_POS', 'data', 'wiki2')
VAL = os.path.join(DATA, 'val_original_10_freq.txt')
VOCAB = os.path.join(DATA, 'vocab_file_10_freq.txt')


class _Captured(Exception):
    pass


def _capture_starts(module, argv, entry):
    """Run ``entry(argv)`` until its first validation and return the window starts."""
    real = module.evaluate
    seen = {}

    def fake(model, dataset, starts, *args, **kwargs):
        seen['starts'] = np.array(starts, copy=True)
        seen['length'] = dataset.length
        raise _Captured()

    module.evaluate = fake
    try:
        entry(argv)
    except _Captured:
        pass
    finally:
        module.evaluate = real
    return seen


def test_validation_windows_match_lut_torch():
    with tempfile.TemporaryDirectory() as tmp:
        common = [VAL, '--validation-data', VAL, '--vocab-file', VOCAB,
                  '--context-size', '32', '--batch-size', '2', '--max-steps', '2',
                  '--testing-length', '200', '--seed', '43', '--device', 'cpu']
        lut = _capture_starts(lt, common + [
            '--tokenizer', 'word', '--new-attention', '--embedding-dim', '16',
            '--num-layers', '1', '--num-heads', '1', '--n-t', '2', '--n-c', '3',
            '--attn_qk_dim', '4', '--attn_qk_n_c', '3', '--attn_qk_n_t', '2',
            '--warmup-steps', '1', '--cpu-cores', '1', '--cuda-graphs', 'off',
            '--loss-file', os.path.join(tmp, 'lut.csv')], lt.main)
        gpt = _capture_starts(gbt, common + [
            '--n-layer', '1', '--n-head', '1', '--n-embd', '16', '--warmup-steps', '1',
            '--cpu-cores', '1', '--loss-file', os.path.join(tmp, 'gpt.csv')], gbt.main)
    assert lut['length'] == gpt['length']
    assert len(lut['starts']) == 200
    np.testing.assert_array_equal(lut['starts'], gpt['starts'])


def test_full_logits_match_nanogpt_forward():
    nanogpt = gbt.load_nanogpt()
    torch.manual_seed(0)
    gpt = nanogpt.GPT(nanogpt.GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8,
                                        vocab_size=50, dropout=0.0, bias=False))
    gpt.eval()
    idx = torch.randint(0, 50, (3, 8))
    expected, _ = gpt(idx, idx)  # with targets, GPT returns logits at every position
    got = gbt.FullLogits(gpt)(idx)
    assert got.shape == (3, 8, 50)
    torch.testing.assert_close(got, expected)


def test_tiny_run_writes_lut_format_loss_csv():
    with tempfile.TemporaryDirectory() as tmp:
        loss_file = os.path.join(tmp, 'loss.csv')
        stem = os.path.join(tmp, 'model')
        gbt.main([VAL, '--validation-data', VAL, '--vocab-file', VOCAB,
                  '--n-layer', '1', '--n-head', '2', '--n-embd', '16',
                  '--batch-size', '4', '--max-steps', '5', '--warmup-steps', '2',
                  '--validation-interval', '2', '--testing-length', '20',
                  '--validation-batch-size', '7', '--device', 'cpu',
                  '--loss-file', loss_file, '--save-model', stem])
        with open(loss_file) as f:
            lines = f.read().strip().splitlines()
        assert lines[0] == ('step, epoch, tokens, loss, perplexity, accuracy, '
                            'loss_no_unk, perplexity_no_unk, accuracy_no_unk, '
                            'loss_per_word, perplexity_per_word')
        rows = [[float(x) for x in line.split(',')] for line in lines[1:]]
        # Validates at steps 0, 2, 4 (interval) -- 4 is also the final step.
        assert [int(r[0]) for r in rows] == [0, 2, 4]
        for r in rows:
            assert len(r) == 11
            assert math.isnan(r[9]) and math.isnan(r[10])  # word tokenizer
            assert abs(r[4] - math.exp(r[3])) / r[4] < 1e-3
        # Near-uniform at initialisation over a 14,398-word vocabulary.
        assert abs(rows[0][3] - math.log(14398)) < 0.5
        assert os.path.exists(stem + '.pt') and os.path.exists(stem + '_best.pt')
        blob = torch.load(stem + '.pt', map_location='cpu', weights_only=False)
        assert blob['kind'] == 'nanogpt_lm' and blob['model_args']['vocab_size'] == 14398


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
