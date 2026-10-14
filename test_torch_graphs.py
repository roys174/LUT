"""CUDA-graph capture must change speed and nothing else.

Skipped entirely without a GPU.  The claims under test are the ones that would fail
silently in a training run rather than raise:

* graphed gradients equal eager gradients, on *consecutive* steps -- this is what proves
  the graph overwrites ``.grad`` rather than accumulating into the previous replay's
  values (see ``lut_torch/graphs.py``: grads are set to None before capture on purpose);
* the reported loss / accuracy statistics match;
* dropout still redraws every replay, rather than freezing the capture-time mask;
* a caller that wrongly calls ``zero_grad(set_to_none=True)`` between replays is caught.

Run with:  .venv/bin/python test_torch_graphs.py
       or: python -m pytest -q test_torch_graphs.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from lut_torch._testing import parametrize, run_all, skip
from lut_torch.config import LutConfig
from lut_torch.data import IGNORE_INDEX
from lut_torch.modules import LutModel

HAS_CUDA = torch.cuda.is_available()

CFG_KWARGS = dict(
    context_size=12, embedding_dim=16, positional_dim=3, num_layers=2, num_heads=3,
    vocab_size_input=61, vocab_size_output=9, n_t=8, n_c=4, attn_qk_dim=3,
    attn_qk_n_c=4, attn_qk_n_t=8, weight_sharing=4, comparisons_per_bit=1,
    qkpe_direct_sign=True, freeze_PE=True, POS_task=True, layernorm=True,
    table_grad='dense',
)


def make(seed=3, dropout=0.0, batch=2, **overrides):
    cfg = LutConfig(dropout=dropout, **{**CFG_KWARGS, **overrides})
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    model = LutModel(cfg).cuda()
    with torch.no_grad():
        gen = torch.Generator(device='cuda').manual_seed(seed)
        for table in model.table_parameters():
            table.copy_(torch.randn(table.shape, generator=gen, device='cuda') * 0.3)
        for w in model.tree_weight_parameters():   # away from 1, or a bug could hide
            w.copy_(torch.rand(w.shape, generator=gen, device='cuda') + 0.5)
    model.train()
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(4):
        inp = torch.from_numpy(
            rng.integers(0, cfg.vocab_size_input, (batch, cfg.context_size))).cuda()
        tgt = rng.integers(0, cfg.vocab_size_output, (batch, cfg.context_size))
        tgt[0, 2] = IGNORE_INDEX                      # exercise the masked path
        batches.append((inp, torch.from_numpy(tgt).cuda()))
    return model, cfg, batches


def eager_step(model, cfg, inp, tgt, batch):
    for p in model.parameters():
        p.grad = None
    logits = model(inp)
    flat = logits.reshape(-1, cfg.vocab_size_output)
    ft = tgt.reshape(-1)
    total = F.cross_entropy(flat, ft, ignore_index=IGNORE_INDEX, reduction='sum')
    (total / batch).backward()
    valid = ft != IGNORE_INDEX
    count = int(valid.sum())
    correct = int(((flat.argmax(dim=-1) == ft) & valid).sum())
    grads = {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in model.named_parameters()}
    return float(total.detach()) / count, correct, count, grads


@parametrize('tree_weights', [False, True])
def test_graph_matches_eager_over_consecutive_steps(tree_weights):
    """The property that catches accumulate-instead-of-overwrite: it would look right on
    replay 1 and drift from replay 2 onwards."""
    if not HAS_CUDA:
        skip('no CUDA device')
    from lut_torch.graphs import GraphedTrainStep

    batch = 2
    model_e, cfg, batches = make(batch=batch, tree_weights=tree_weights)
    expected = [eager_step(model_e, cfg, i, t, batch) for i, t in batches]

    model_g, _, batches_g = make(batch=batch, tree_weights=tree_weights)
    graphed = GraphedTrainStep(model_g, batches_g[0][0], batches_g[0][1],
                               cfg.vocab_size_output, IGNORE_INDEX, batch)
    for step, ((inp, tgt), (want_loss, want_correct, want_count, want_grads)) in enumerate(
            zip(batches_g, expected)):
        loss, correct, count = graphed.run(inp, tgt)
        assert count == want_count, f'step {step}: count {count} != {want_count}'
        assert correct == want_correct, f'step {step}: correct {correct} != {want_correct}'
        assert abs(loss - want_loss) < 1e-4, f'step {step}: loss {loss} != {want_loss}'
        for name, param in model_g.named_parameters():
            want = want_grads[name]
            if want is None:
                continue
            got = param.grad
            assert got is not None, f'step {step}: {name} has no gradient'
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5,
                                       msg=f'step {step}, {name}')


@parametrize('tree_weights', [False, True])
def test_optimizer_steps_apply_through_the_graph(tree_weights):
    """A full loop -- replay, opt.step(), replay -- must track the eager equivalent.

    Catches the case where gradients land somewhere the optimizer never reads: training
    would run at full speed and simply not learn.
    """
    if not HAS_CUDA:
        skip('no CUDA device')
    from lut_torch.graphs import GraphedTrainStep

    batch = 2
    model_e, cfg, batches = make(batch=batch, tree_weights=tree_weights)
    opt_e = torch.optim.SGD(list(model_e.parameters()), lr=0.05)
    for inp, tgt in batches:
        eager_step(model_e, cfg, inp, tgt, batch)
        opt_e.step()

    model_g, _, batches_g = make(batch=batch, tree_weights=tree_weights)
    opt_g = torch.optim.SGD(list(model_g.parameters()), lr=0.05)
    graphed = GraphedTrainStep(model_g, batches_g[0][0], batches_g[0][1],
                               cfg.vocab_size_output, IGNORE_INDEX, batch)
    for inp, tgt in batches_g:
        graphed.run(inp, tgt)
        opt_g.step()

    for (name, a), (_, b) in zip(model_e.named_parameters(), model_g.named_parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=name)


def test_dropout_redraws_every_replay():
    if not HAS_CUDA:
        skip('no CUDA device')
    from lut_torch.graphs import GraphedTrainStep

    model, cfg, batches = make(dropout=0.5, batch=2)
    graphed = GraphedTrainStep(model, batches[0][0], batches[0][1],
                               cfg.vocab_size_output, IGNORE_INDEX, 2)
    seen = []
    for _ in range(3):
        graphed.run(*batches[0])                       # identical batch every time
        seen.append(model.output_head.S_flat.grad.clone())
    assert not torch.equal(seen[0], seen[1]), 'dropout mask frozen at capture time'
    assert not torch.equal(seen[1], seen[2]), 'dropout mask frozen at capture time'


def test_set_to_none_between_replays_is_caught():
    if not HAS_CUDA:
        skip('no CUDA device')
    from lut_torch.graphs import GraphedTrainStep

    model, cfg, batches = make(batch=2)
    graphed = GraphedTrainStep(model, batches[0][0], batches[0][1],
                               cfg.vocab_size_output, IGNORE_INDEX, 2)
    graphed.run(*batches[0])
    graphed.assert_grads_intact()                      # the correct usage passes
    torch.optim.SGD(list(model.parameters()), lr=0.1).zero_grad(set_to_none=True)
    try:
        graphed.assert_grads_intact()
    except RuntimeError as exc:
        assert 'zero_grad' in str(exc)
    else:
        raise AssertionError('freeing captured gradient buffers went undetected')


if __name__ == '__main__':
    raise SystemExit(run_all(globals()))
