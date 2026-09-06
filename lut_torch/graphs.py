"""CUDA-graph capture of the training step.

At batch 1 this model is not compute-bound, it is *launch*-bound, twice over.  Measured on
an RTX 2080 Ti at the production POS shape (CS 32, ED 64, 6 layers, 8 heads, n_t 32,
n_c 8): one step issues ~29,000 CUDA kernel launches, of which only ~94 ms of the 373 ms
wall is GPU-busy -- and that 94 ms is itself ~29,000 kernels at this card's ~3.2 us
minimum kernel duration, so almost none of it is arithmetic.  The other ~280 ms is Python
plus autograd plus ATen dispatch plus ``cudaLaunchKernel``, at ~12 us a launch.

A CUDA graph records that whole kernel sequence once and replays it from a single launch,
which deletes the CPU-side cost outright.  It works here because the step is completely
static: fixed shapes, no data-dependent control flow, no host syncs inside the step.

**Only forward and backward are captured.**  The optimizer and the LR scheduler stay
eager, deliberately:

* they are a handful of ``foreach`` launches (``SGD.step`` profiles as a single
  ``_foreach_add_``), so capturing them buys essentially nothing -- measured, capturing
  fwd+bwd alone gives 96.2 ms and adding the optimizer to the capture gives 110.6 ms,
  which is what the eager optimizer costs anyway;
* the LR schedule writes a Python float into ``param_groups`` every step.  A captured
  ``opt.step()`` would bake the learning rate in as a constant at capture time and the
  schedule would silently stop taking effect -- a bug that shows up only as a worse loss
  curve.  Leaving the optimizer eager makes that impossible;
* ``SparseAdam``, checkpoint save/load and ``--momentum`` then keep working exactly as
  they do without graphs.

Numerics are unchanged.  Capture does not alter which kernels run, in what order, or on
what data; it only removes the CPU work of launching them.  Dropout still redraws every
replay (torch's graph-safe philox offset).  ``test_torch_graphs.py`` checks both claims.
"""

import torch
import torch.nn.functional as F


class GraphedTrainStep:
    """Forward + backward for one fixed-shape batch, replayed from a single launch.

    Usage::

        graphed = GraphedTrainStep(model, inputs, targets, vocab, ignore_index, B)
        for step in ...:
            mean_loss, correct, count = graphed.run(inputs, targets)
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

    Note what is *absent*: there is no ``zero_grad``.  Gradients are set to ``None``
    immediately before capture, so ``backward()`` allocates each ``.grad`` inside the
    graph's private memory pool and *assigns* into it rather than accumulating.  Every
    replay then overwrites those same buffers, which both removes the need to zero and
    saves the zero-fill itself -- 0.93 GB per step at the production shape.

    The caller must therefore **never** call ``opt.zero_grad(set_to_none=True)`` between
    replays: that frees the graph-pool tensors while the graph keeps writing to their
    addresses, leaving the optimizer looking at ``None``.  :meth:`assert_grads_intact`
    exists to catch that mistake loudly.

    The overwrite-instead-of-accumulate behaviour assumes each parameter receives exactly
    one gradient contribution per step, which holds here (no table is shared between two
    LUT calls).  ``test_torch_graphs.py`` verifies it by comparing graphed gradients
    against eager ones over several consecutive steps.
    """

    def __init__(self, model, inputs, targets, vocab_size, ignore_index, batch_size,
                 warmup=3):
        """``inputs``/``targets`` are a representative batch, used for warmup and capture.

        They are cloned into the static buffers :meth:`run` copies into, so the caller
        keeps ownership of what it passed.
        """
        if inputs.device.type != 'cuda':
            raise RuntimeError('CUDA graphs require CUDA tensors')
        self.model = model
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.static_inputs = inputs.detach().clone()
        self.static_targets = targets.detach().clone()

        # Warmup is mandatory before capture -- it forces autograd to build its graph
        # once, lets the caching allocator settle, and must run on a side stream.  It
        # burns real RNG, so the generator states are saved and put back afterwards: a
        # graphed run then consumes the same RNG stream as an eager one from the same
        # seed instead of starting several dropout draws ahead.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._grads_to_none()
                self._build()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

        # Grads to None so backward allocates them from the graph's pool (see class doc).
        self._grads_to_none()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_stats = self._build()
        # Capture records without executing, so every .grad is allocated but holds
        # garbage until the first replay.  Remember them, to detect a caller that frees
        # them behind our back.
        self._grad_ptrs = {name: (p.grad.data_ptr() if p.grad is not None else None)
                           for name, p in model.named_parameters() if p.requires_grad}

    def _grads_to_none(self):
        for param in self.model.parameters():
            param.grad = None

    def _build(self):
        """The captured region: forward, loss, backward, and the stats to read back.

        Returns one stacked tensor rather than three so :meth:`run` pays a single
        device-to-host sync per step instead of three.
        """
        logits = self.model(self.static_inputs)
        flat = logits.reshape(-1, self.vocab_size)
        flat_targets = self.static_targets.reshape(-1)
        total = F.cross_entropy(flat, flat_targets, ignore_index=self.ignore_index,
                                reduction='sum')
        (total / self.batch_size).backward()
        with torch.no_grad():
            valid = flat_targets != self.ignore_index
            count = valid.sum()
            correct = ((flat.argmax(dim=-1) == flat_targets) & valid).sum()
            return torch.stack([total.detach(), count.to(total.dtype),
                                correct.to(total.dtype)])

    def assert_grads_intact(self):
        """Raise if a ``.grad`` the graph writes into was freed or replaced.

        The failure this guards against is silent otherwise: the graph keeps writing to
        the captured addresses while the optimizer reads ``None`` (or a different, never
        updated buffer), so training simply stops making progress.
        """
        for name, ptr in self._grad_ptrs.items():
            grad = dict(self.model.named_parameters())[name].grad
            if ptr is None:
                continue
            if grad is None or grad.data_ptr() != ptr:
                raise RuntimeError(
                    f"gradient buffer for '{name}' was replaced after CUDA-graph capture "
                    f"-- do not call zero_grad(set_to_none=True) between replays; the "
                    f"graph overwrites gradients on every replay, so no zeroing is needed")

    def run(self, inputs, targets):
        """Replay the step on a new batch.  Returns ``(mean_loss, correct, count)``.

        Gradients land in the same ``.grad`` tensors the eager path would fill, so the
        caller's ``opt.step()`` needs no changes.
        """
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_targets.copy_(targets, non_blocking=True)
        self.graph.replay()
        total, count, correct = self.static_stats.tolist()
        count = int(count)
        return (total / count if count else 0.0), int(correct), count
