"""Tests for the low-dimensional coded output head (``--output-code-dim``).

The head emits ``D`` numbers instead of one score per class, and a ``(V, D)`` code matrix
maps them to the logits.  Three things are worth pinning down:

* the default path is untouched -- same shapes, same tables, same RNG consumption, so
  every existing result and the parity gate stay valid;
* the coded path actually shrinks the table and still trains both the table and (when
  learned) the code;
* the linear readout is not an approximation of the per-bit decoder the head is meant to
  implement -- it is algebraically the same thing (:func:`test_bit_likelihood_is_linear`).

This flag has no ``main.py`` counterpart, so it cannot enter ``test_torch_parity.CONFIGS``;
:func:`test_default_head_unchanged` is what stands in for that.

Run with:  .venv/bin/python test_torch_output_code.py
       or: python -m pytest -q test_torch_output_code.py
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lut_torch._testing import parametrize, run_all
from lut_torch.config import ConfigError, LutConfig
from lut_torch.modules import (LutModel, binary_code_matrix, ecoc_masks,
                               random_code_matrix)
from lut_torch.train import build_optimizers, build_parser, lr_lambda_factory

CFG_KWARGS = dict(
    context_size=8, embedding_dim=16, positional_dim=3, num_layers=2, num_heads=2,
    vocab_size_input=23, vocab_size_output=18, n_t=4, n_c=3, attn_qk_dim=4,
    attn_qk_n_c=3, attn_qk_n_t=3, POS_task=True, new_MHA=True,
)


def build(seed=7, **extra):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return LutModel(LutConfig(**CFG_KWARGS, **extra))


def forward_backward(model, seed=0):
    """One CE step. Returns the logits and the output table's gradient (dense)."""
    gen = torch.Generator().manual_seed(seed)
    cfg = model.cfg
    tokens = torch.randint(0, cfg.vocab_size_input, (3, cfg.context_size),
                           generator=gen)
    targets = torch.randint(0, cfg.vocab_size_output, (3, cfg.context_size),
                            generator=gen)
    logits = model(tokens)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size_output),
                    targets.reshape(-1)).backward()
    grad = model.output_head.S_flat.grad
    return logits, grad.to_dense() if grad.is_sparse else grad


def test_default_head_unchanged():
    """No code -> the old head, and the new draw must not perturb the RNG streams."""
    model = build()
    assert model.output_code is None
    assert model.output_head.y_dim == CFG_KWARGS['vocab_size_output']
    logits, _ = forward_backward(model)
    assert logits.shape == (3, CFG_KWARGS['context_size'],
                            CFG_KWARGS['vocab_size_output'])

    # The code matrix is drawn last and only when asked for, so turning the flag on must
    # leave every other tensor in the model exactly where it was. If this fails, the new
    # draw has moved an RNG stream and every existing seeded result has shifted with it.
    coded = build(output_code_dim=5, output_code='learned').state_dict()
    plain = model.state_dict()
    assert set(coded) - set(plain) == {'output_code'}
    for key, value in plain.items():
        if key == 'output_head.S_flat':
            continue  # legitimately narrower under a code
        assert torch.equal(value, coded[key]), key


@parametrize('mode', ['learned', 'binary'])
def test_coded_head_shapes_and_gradients(mode):
    code_dim = 5
    model = build(output_code_dim=code_dim, output_code=mode)
    vocab = CFG_KWARGS['vocab_size_output']

    assert model.output_head.y_dim == code_dim
    assert model.output_code.shape == (vocab, code_dim)
    # The table is the whole point: D columns instead of V.
    assert model.output_head.S_flat.shape[-1] == code_dim
    assert (model.output_head.S_flat.numel()
            == build().output_head.S_flat.numel() * code_dim // vocab)

    is_learned = mode == 'learned'
    assert isinstance(model.output_code, nn.Parameter) is is_learned
    # Identity, not `in`: `==` on tensors returns a tensor, not a bool. A learned C
    # belongs to the output-code group, never to dense -- that split is what
    # --output-code-lr-scale acts on.
    assert any(p is model.output_code
               for p in model.output_code_parameters()) is is_learned
    assert not any(p is model.output_code for p in model.dense_parameters())
    assert ('output_code' in dict(model.named_buffers())) == (not is_learned)

    logits, table_grad = forward_backward(model)
    assert logits.shape == (3, CFG_KWARGS['context_size'], vocab)
    # A zero table gradient would make the whole comparison vacuous -- it is the
    # failure mode a zero-initialised code matrix would cause.
    assert float(table_grad.abs().sum()) > 0.0

    if is_learned:
        # dL/dC = g (x) y, and y is the sum of table rows, which are zero-initialised --
        # so on the FIRST step C necessarily receives no gradient at all, whatever C is.
        # That is not a defect: grad_y = C.T g is already non-zero (which is why C must
        # not start at zero), so the tables move, and from the next step y != 0 and C
        # trains. Asserting otherwise here is what made this test fail.
        assert float(model.output_code.grad.abs().sum()) == 0.0
        torch.optim.SGD(model.table_parameters(), lr=0.5).step()
        model.zero_grad(set_to_none=True)
        forward_backward(model, seed=1)
        assert float(model.output_code.grad.abs().sum()) > 0.0


@parametrize('mode', ['learned', 'binary'])
def test_bias_is_separate_and_trains(mode):
    vocab = CFG_KWARGS['vocab_size_output']
    model = build(output_code_dim=5, output_code=mode, output_code_bias=True)
    assert model.output_bias.shape == (vocab,)
    # Zero-initialised, so turning the bias on cannot shift the tables it shares a seed
    # with -- it draws no RNG.
    assert float(model.output_bias.abs().sum()) == 0.0

    code_params = model.output_code_parameters()
    assert any(p is model.output_bias for p in code_params)
    assert (any(p is model.output_code for p in code_params)) is (mode == 'learned')
    # C and b must be out of the dense group, or --output-code-lr-scale would silently
    # do nothing to them.
    dense_ids = {id(p) for p in model.dense_parameters()}
    assert all(id(p) not in dense_ids for p in code_params)

    forward_backward(model)
    assert float(model.output_bias.grad.abs().sum()) > 0.0


def test_bias_shifts_logits_by_itself():
    model = build(output_code_dim=5, output_code='binary', output_code_bias=True)
    tokens = torch.randint(0, CFG_KWARGS['vocab_size_input'],
                           (2, CFG_KWARGS['context_size']))
    with torch.no_grad():
        before = model(tokens)
        model.output_bias.add_(torch.arange(float(CFG_KWARGS['vocab_size_output'])))
        after = model(tokens)
    # An additive per-class term, identical at every position -- that is the whole point:
    # it is the one part of the logits that does not have to live in col(C).
    assert torch.allclose(after - before,
                          torch.arange(float(CFG_KWARGS['vocab_size_output'])).expand_as(
                              before), atol=1e-5)


def test_binary_code_widens_into_an_ecoc():
    """Above ceil(log2 V) the extra bits are parity checks, not padding.

    Zero-padding the index expansion would give constant columns, which shift every logit
    equally and cancel in the softmax -- no separation bought. The systematic ECOC adds a
    real coordinate per bit, and minimum Hamming distance is what governs how many
    coordinate sign errors the decode survives.
    """
    for width, distinct_cols in ((5, 5), (10, 10), (18, 18)):
        code = binary_code_matrix(18, width)
        assert code.shape == (18, width)
        signs = (code > 0).to(torch.int64)
        assert len({tuple(r.tolist()) for r in signs}) == 18
        # Every column must carry information -- this is exactly what padding failed.
        assert sum(len(signs[:, c].unique()) == 2 for c in range(width)) == distinct_cols

    # D = ceil(log2 18) reproduces the plain index expansion, so results measured at that
    # width before the ECOC generalisation still stand.
    idx = torch.arange(18)
    ref = ((idx[:, None] >> torch.arange(5)[None, :]) & 1)
    assert torch.equal((binary_code_matrix(18, 5) > 0).to(torch.int64), ref)

    def min_distance(code):
        s = (code > 0).to(torch.int64)
        d = (s[:, None, :] != s[None, :, :]).sum(-1)
        iu = torch.triu_indices(18, 18, offset=1)
        return int(d[iu[0], iu[1]].min())

    # Redundant bits must actually buy separation, and beat an undrawn-by-design code.
    assert min_distance(binary_code_matrix(18, 5)) == 1
    assert min_distance(binary_code_matrix(18, 12)) >= 3
    assert min_distance(binary_code_matrix(18, 18)) >= 5
    assert min_distance(binary_code_matrix(18, 18)) > min_distance(
        random_code_matrix(18, 18))


@parametrize('code_dim', [5, 8, 12, 18])
def test_random_code_is_frozen_distinct_and_full_rank(code_dim):
    model = build(output_code_dim=code_dim, output_code='random')
    code = model.output_code
    assert code.shape == (18, code_dim)
    assert 'output_code' in dict(model.named_buffers())   # frozen, never a parameter
    assert not any(p is code for p in model.output_code_parameters())
    assert torch.allclose(code.abs(), torch.full_like(code, 1 / math.sqrt(code_dim)))
    # Distinct codewords, and every column carries information -- the two properties the
    # binary code loses when widened.
    signs = (code > 0).to(torch.int64)
    assert len({tuple(row.tolist()) for row in signs}) == 18
    assert all(len(signs[:, c].unique()) == 2 for c in range(code_dim))

    # Separation is the reason for the extra bits: a random code's expected Hamming
    # distance is D/2, against the binary code's minimum of 1.
    ham = (signs[:, None, :] != signs[None, :, :]).sum(-1)
    iu = torch.triu_indices(18, 18, offset=1)
    assert int(ham[iu[0], iu[1]].min()) >= 1
    if code_dim >= 8:
        assert float(ham[iu[0], iu[1]].float().mean()) > code_dim / 3


def test_bias_requires_a_code():
    try:
        build(output_code_bias=True)
    except ConfigError:
        return
    raise AssertionError('expected ConfigError for a bias without a code')


def test_frozen_code_does_not_move():
    model = build(output_code_dim=5, output_code='binary')
    before = model.output_code.clone()
    forward_backward(model)
    opt = torch.optim.SGD(model.dense_parameters(), lr=1.0)
    opt.step()
    assert torch.equal(model.output_code, before)


def test_scale_one_matches_a_single_dense_group():
    """At scale 1.0 the split into a separate param group must change nothing.

    This is what makes the round-1 results (baseline 0.9449 / learned:5 0.9429 /
    binary:5 0.9359 at seed 42) still comparable with anything launched after the split,
    rather than costing three GPU-hours to re-measure.
    """
    args = build_parser().parse_args(['data.txt'])
    # Past the start of warmup, so the steps below are taken at a non-zero rate and the
    # comparison can actually detect a difference.
    args.max_steps, args.max_learning_rate, args.warmup_steps = 1000, 0.04, 1
    lr = args.max_learning_rate

    def train(split):
        model = build(output_code_dim=5, output_code='learned', output_code_bias=True)
        if split:
            opts, scheds = build_optimizers(model, args)
        else:  # the pre-split grouping: C and b sat in with PE and the LayerNorm affines
            params = model.dense_parameters() + model.output_code_parameters()
            opts = [torch.optim.SGD([{'params': model.table_parameters()}], lr=lr),
                    torch.optim.SGD([{'params': params}], lr=lr)]
            # The same LambdaLR the split path gets. Without it the two branches take
            # steps at different learning rates -- LambdaLR zeroes the rate at
            # construction for the start of warmup -- and the comparison is meaningless.
            factor = lr_lambda_factory(args)
            scheds = [torch.optim.lr_scheduler.LambdaLR(o, factor) for o in opts]
        for step in range(3):
            for opt in opts:
                opt.zero_grad()
            forward_backward(model, seed=step)
            for opt in opts:
                opt.step()
            for sched in scheds:
                sched.step()
        return model.state_dict()

    split, single = train(True), train(False)
    for key, value in single.items():
        assert torch.equal(value, split[key]), key


@parametrize('scale', [1.0, 8.0, 64.0])
def test_lr_scale_reaches_only_the_code_group(scale):
    """The scaled group must contain C and b, and nothing else must be scaled."""
    model = build(output_code_dim=5, output_code='learned', output_code_bias=True)
    parser = build_parser()
    args = parser.parse_args(['data.txt', '--output-code-lr-scale', str(scale)])
    args.max_steps, args.max_learning_rate = 1000, 0.04
    optimizers, schedulers = build_optimizers(model, args)

    groups = [g for opt in optimizers for g in opt.param_groups]
    code_ids = {id(p) for p in model.output_code_parameters()}
    scaled = [g for g in groups if code_ids & {id(p) for p in g['params']}]
    assert len(scaled) == 1
    assert {id(p) for p in scaled[0]['params']} == code_ids

    # Assert on initial_lr, not lr: LambdaLR applies factor(step 0) inside its own
    # constructor, and step 0 is the start of warmup, so every group's live `lr` is
    # already 0.0 by the time build_optimizers returns. initial_lr is the peak the
    # schedule multiplies, which is what --output-code-lr-scale actually sets.
    assert scaled[0]['initial_lr'] == args.max_learning_rate * scale
    for g in groups:
        if g is not scaled[0]:
            assert g['initial_lr'] == args.max_learning_rate

    # And the ratio has to survive the schedule, not just its construction.
    for _ in range(args.warmup_steps):
        for sched in schedulers:
            sched.step()
    others = [g for g in groups if g is not scaled[0]]
    assert scaled[0]['lr'] > 0.0
    assert abs(scaled[0]['lr'] - others[0]['lr'] * scale) < 1e-12


def test_binary_code_is_the_class_index():
    code = binary_code_matrix(18, 5)
    assert code.shape == (18, 5)
    assert torch.allclose(code.abs(), torch.full_like(code, 1 / math.sqrt(5)))
    # Distinct codewords, and each one really is the class index in binary.
    for k in range(18):
        bits = ((code[k] > 0).to(torch.int64)
                * (2 ** torch.arange(5, dtype=torch.int64))).sum()
        assert int(bits) == k


def test_bit_likelihood_is_linear():
    """``y @ C.T`` *is* the per-bit likelihood decoder, not a relaxation of it.

    Scoring class ``k`` by ``sum_b [c_kb log sigma(y_b) + (1 - c_kb) log sigma(-y_b)]``
    equals ``sum_b (2 c_kb - 1) y_b / 2`` plus a term independent of ``k``, which the
    softmax removes.  Verified here so the docstring in ``LutModel.forward`` cannot
    quietly stop being true.
    """
    torch.manual_seed(0)
    vocab, code_dim = 18, 5
    signed = binary_code_matrix(vocab, code_dim) * math.sqrt(code_dim)
    bits01 = (signed + 1) / 2
    y = torch.randn(11, code_dim) * 3.0

    per_bit = (bits01 @ F.logsigmoid(y).T + (1 - bits01) @ F.logsigmoid(-y).T).T
    linear = y @ signed.T / 2.0
    assert torch.allclose(F.log_softmax(per_bit, dim=-1),
                          F.log_softmax(linear, dim=-1), atol=1e-5)


@parametrize('kwargs', [
    dict(output_code_dim=0),
    dict(output_code_dim=19),
    dict(output_code_dim=4, output_code='binary'),   # needs ceil(log2(18)) = 5
    dict(output_code_dim=5, output_code='nonsense'),
])
def test_rejected_configurations(kwargs):
    try:
        build(**kwargs)
    except ConfigError:
        return
    raise AssertionError(f'expected ConfigError for {kwargs}')


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))


def test_tree_chunking_is_off_at_normal_widths():
    """The chunked backward must be the single-call path for everything but a wide head.

    That is what keeps `test_torch_parity.py` meaningful: at POS widths this is not an
    approximation of the old code, it is literally the same call.
    """
    from lut_torch import functional as fn
    # (n_trees, lead = B*P, y_dim) for the production POS shape at batch 32.
    assert fn._tree_chunk(64, 32 * 32, 18) == 64
    assert fn._tree_chunk(64, 32 * 32, 64) == 64
    # A full vocabulary does not fit, so it splits.
    assert fn._tree_chunk(64, 32 * 32, 14398) == 1
    assert fn._tree_chunk(64, 32 * 32, 512) < 64


@parametrize('mode', ['learned', None])
def test_chunked_backward_matches_unchunked(mode):
    """Forcing a tiny budget must not change any gradient."""
    from lut_torch import functional as fn

    def grads(budget):
        saved = fn.GATHER_BUDGET_ELEMS
        fn.GATHER_BUDGET_ELEMS = budget
        try:
            extra = {} if mode is None else dict(output_code_dim=5, output_code=mode)
            model = build(**extra)
            forward_backward(model)
            out = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    g = p.grad
                    out[name] = (g.to_dense() if g.is_sparse else g).clone()
            return out
        finally:
            fn.GATHER_BUDGET_ELEMS = saved

    big, small = grads(1 << 24), grads(1)
    assert set(big) == set(small)
    for name, value in big.items():
        # Chunking reassociates one scatter-add, so float32 may differ in the last ULP.
        assert torch.allclose(value, small[name], rtol=1e-5, atol=1e-6), name
