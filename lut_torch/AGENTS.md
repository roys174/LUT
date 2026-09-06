# AGENTS.md — `lut_torch/`

Read this before changing anything in this directory. The repository-wide file at
`../AGENTS.md` still governs cluster/sandbox rules, the data and POS cache, `main.py`, and
the analysis tooling; this file governs the torch codebase.

This is research code. Preserve experiment provenance, prefer small testable changes, and
treat the parity suite as a contract rather than a formality.

## What this is

`lut_torch/` is a PyTorch implementation of the LUT (look-up table) model — the
`--new-attention` / `--new-MHA` architecture — for POS tagging and plain next-token LM.
It exists so that torch owns autograd, optimizers, schedulers and checkpointing, leaving
us to own only the part that is genuinely custom: how a LUT lookup differentiates.

`main.py` is untouched and remains the reference implementation. It still runs everything
this package deliberately dropped.

## Start here

| File | What lives in it |
|---|---|
| `functional.py` | **The only custom autograd.** `LUTFunction`, `lut_lookup`, `LutGroup`, and the pure-autograd oracle `lut_lookup_reference`. Read this first. |
| `modules.py` | `LUTTable`, `NewAttentionHead`, `LutLayer`, `LutModel`. Construction order mirrors `main.build_Model` exactly. |
| `comparisons.py` | a/b comparison-pair construction and PE init, RNG-order-faithful to `main.py`. |
| `config.py` | `LutConfig`: structural flags, validation, derived fields. |
| `data.py` | Tokenizer/POS loading (shares `main.py`'s cache filenames) and window batching. |
| `train.py` | CLI trainer: loop, validation, optimizers, schedule, checkpoints, W&B. |
| `bench_cpu.py` | CPU scaling benchmark (`python -m lut_torch.bench_cpu`). |
| `_testing.py` | pytest shim so tests run on either interpreter here. |

Tests live at the repository root: `test_torch_lut_primitive.py`, `test_torch_parity.py`,
`test_torch_training.py`.

## The core idea

A LUT turns a real vector into a discrete row index and reads a table:

```
bit_b = XOR_k ( u[..., b*K + k] > 0 )     # u = x[a] - x[b], K = comparisons_per_bit
j     = sum_b bit_b << b                  # per group, shifted by the group's bit_offset
y[p]  = sum_t S[trees[t], j[p, t], :]
```

Nothing there is differentiable, so `main.py` defines two gradients by hand. Both fit torch:

1. **Table gradient.** `S[tree, j] -= lr * grad`, accumulated over every (position, tree)
   that landed in the bin, *is* the backward of an embedding lookup — a scatter-add. Its
   representation is chosen by `--table-grad` (see "Table gradients" below); do not assume
   sparse.

2. **Input gradient.** The surrogate `v = gi * (-0.5*sign(u)/(1+|u|)^2)` is exactly
   `dphi/du` for `phi(u) = 0.5/(1+|u|)`. (`phi` is not a smooth version of the bit and is
   not monotonic; it is simply the antiderivative of the chosen surrogate, which is all
   that is needed.)

**The load-bearing design decision:** the custom step is defined over the *margins* `u`,
not the layer input `x`. Everything upstream — `u = x[a] - x[b]`, QK normalisation, PE
addition, direct-sign slicing — is then ordinary torch, and autograd derives all the
routing that `main.new_attention_backward` writes out by hand
(`_normalize_vector_backward`, and the `pe_grad`/`q_grad`/`k_grad` scattering for both the
direct-sign and a/b layouts). `--freeze-PE` becomes `requires_grad=False`; `--PE-only`
becomes a zeroed term in the forward.

Do not "simplify" the primitive to take `x` directly. That would drag every one of those
hand-written routing rules back into the codebase.

### Why an explicit `autograd.Function`

Because `phi` is a real function, the whole thing can be written in pure autograd:

```python
y = embedding(j) + (S[jbar] - S[j]).detach() * (phi(u) - phi(u).detach())
```

— zero in the forward pass, exactly the right gradient in backward. That version exists as
`lut_lookup_reference` and is used as a test oracle.

It is **not** the primary implementation because `--gradient-pair-selection cosine-distance`
scores candidate comparisons using the *upstream gradient*, so selection can only happen
inside backward. A pure-autograd formulation cannot express it.

### Groups

A `LutGroup` is one contiguous run of row bits fed by one margin source.

- Simple tables (FFN, Q, K, V, MHA_O, output) have exactly one group.
- The `Final` table has two: `j_final = (v_bits << positional_dim) | qkpe_bits`, i.e. the
  QKPE group at `bit_offset=0` and the V group at `bit_offset=positional_dim`. Each group
  selects its own comparison in backward and routes its gradient to its own source.
- `pair_index` lets a per-*position* margin tensor feed a per-*pair* table without the
  caller materialising the expanded tensor. The V-side margins are per position (CS of
  them); the table is indexed per causal pair (CS·(CS−1)/2).
- Margin tensors are `(B, P_src, N_T, n_bits*fan_in)`; a tree axis of size 1 means the
  margins are shared across trees and will broadcast. `--qkpe-direct-sign` uses that, and
  it is a large memory saving.

## Invariants you can break silently

These are the things that will not raise an error but will quietly change results.

1. **RNG draw order.** `comparisons.py` reproduces `main.build_Model`'s exact draw
   sequence: Python `random` for comparison pairs, NumPy for the embedder and PE,
   interleaved in the original order. The loops look naïve on purpose — vectorising them
   changes the stream and breaks parity. `test_construction_matches` guards this.

2. **Loss scaling.** `main.py` *reports* a mean over valid positions but its table gradient
   is the *sum* over positions (`compute_gradients` writes `softmax - onehot` with no
   `1/CS`), averaged over the batch (replay scales by `lr/bs`). So the backward objective
   is `cross_entropy(..., reduction='sum') / batch_size`, with the mean computed separately
   for logging. Switching to a mean objective silently rescales the effective LR by the
   context size.

3. **Table gradient representation and the optimizer must agree.** Plain SGD accepts
   either; Adam/AdamW cannot consume a sparse gradient, and `SparseAdam` cannot consume a
   dense one. `build_optimizers` is the single point that enforces this (selecting Adam
   pins the tables to dense) — keep it there rather than in argument parsing, which a
   direct caller bypasses. Momentum and weight decay apply to dense parameters (PE,
   LayerNorm) only.

4. **First-minimum tie-breaking.** Closest-comparison selection must return the *first*
   minimum, matching `main.py`'s strict `<` scan and numpy's `argmin`.
   `torch.min(dim=-1).indices` documents this; `torch.argmin` does not.

5. **The token embedding is frozen** (a buffer, not a parameter) because `main.py` never
   updates it — the embedder update is commented out in `model_backward`.

## Table gradients

A LUT's gradient touches one row per **visit** — one per (batch, position-or-pair, tree).
Sparse only pays off while visits are fewer than the table's rows, and for the Final table
the opposite holds: it is indexed per causal pair per tree, so at context 32 one batch
element already makes 496 x 64 = 31,744 visits against 16,384 rows.

Getting this wrong is expensive and does not look like a bug — it looks like bad scaling.
An earlier version returned sparse gradients unconditionally, which made
`optimizer.step()` the single most expensive phase of training (780 ms at batch 8, against
137 ms forward and 673 ms backward) and blew batch 32 up to **368 seconds per step**,
24x worse than linear. Switching to dense took the same step to 19.3 s.

| | nnz (visits) | sparse grad | vs the 4.2 MB dense table |
|---|---|---|---|
| batch 1 | 31,744 | 8.1 MB | 1.9x |
| batch 8 | 253,952 | 65 MB | 15.5x |
| batch 32 | 1,015,808 | 260 MB | 62x |

`--table-grad auto` (the default) picks dense once visits reach the table's row count,
which is the right rule in both directions — a genuinely large, rarely-visited table still
gets sparse. Measured head to head on the Final table, dense wins at every batch size
including batch 1 (1.8x at B=1, 3.4x at B=32). Force `sparse` only when a dense gradient
will not fit.

## How CPUs are used

This is genuinely different from `main.py` and worth understanding before sizing a job.

`main.py` had four parallelism knobs: `--num-workers` (thread pool over batch elements),
`--replay-workers` (thread pool over independent `S` tables during sparse replay),
`--head-workers` (thread pool over MHA heads at batch 1), and `--numba-threads` (`prange`
inside each kernel). **All four are gone.**

`lut_torch` has exactly one: **ATen's intra-op thread pool**, set by
`torch.set_num_threads(args.cpu_cores)`. Each individual tensor operation — the
embedding gathers, the mat-vec contractions, the elementwise margin arithmetic, the
scatters — splits its own work across that pool. There is no parallelism *between*
operations: the Python-level loop over `num_layers x num_heads` runs one op at a time.

Four consequences:

- **Batch size, not thread count, is what makes cores usable.** A thread pool only helps
  when each op has enough work to divide. At batch 1 the per-head tensors are small and
  dispatch overhead dominates, so extra threads buy very little. Raising the batch
  multiplies the work per op while the overhead stays fixed. Threads and batch have to be
  tuned together — that is what `bench_cpu.py --experiment grid` measures.
- **The serial spine is Python dispatch.** Roughly `num_layers x num_heads` head calls per
  step, each a sequence of ops. Threads do not reduce that count, which is why per-sequence
  throughput at batch 1 does not approach the Numba path no matter how many cores you give it.
- **Never oversubscribe.** torch's pool plus a separate OpenMP/MKL pool means threads
  fighting for the same cores. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to the same
  value as `--cpu-cores` (`torch_POS_run.sh` does this).
- **Physical cores, not hyperthreads, and one socket if you can.** The big table gathers
  are memory-bound, so threads past the physical core count mostly add contention, and
  spilling across NUMA sockets adds latency. `torch.set_num_interop_threads` is irrelevant
  here — it applies only to explicitly forked autograd subgraphs, which this model never
  creates.

**Practical rule:** because intra-op scaling is sublinear, `K` independent processes of `T`
threads generally deliver more total throughput than one process of `K*T` threads. That maps
directly onto the multi-seed and ablation workflows — prefer several narrow jobs over one
wide one, and use the wide configuration only when you need a single run to finish sooner.

Memory scales with batch: roughly 0.9 GB at batch 1 and 2.2 GB at batch 16 at the current
POS shape. Size `--mem` accordingly.

### Measured scaling

Context 32, embedding 64, 6 layers, 4 heads, `n_t=64`, `n_c=6`, weight sharing 8, frozen
PE, direct-sign QKPE, no MHA — i.e. the production shape. 20 physical cores, shared host,
so treat these as indicative rather than exact.

| batch | 1 thr | 2 thr | 4 thr | 8 thr | 16 thr |
|---|---|---|---|---|---|
| 1  | 1.80 | 2.66 | 3.29 | 3.39 | 3.37 seq/s |
| 8  | 2.13 | 3.58 | 5.69 | 8.05 | 8.05 seq/s |
| 32 | 1.66 | 2.92 | 5.78 | 9.20 | **11.94 seq/s** |

Parallel efficiency at 4 threads: 46% at batch 1, 67% at batch 8, 87% at batch 32. The
pattern is the point — **batch size buys thread efficiency**. At batch 1 the curve is flat
past 4 threads; at batch 32 it is still climbing at 16.

Splitting a fixed 16-core budget across independent processes (batch 8 each):

| processes x threads | aggregate seq/s | vs one wide job |
|---|---|---|
| 1 x 16 | 8.34 | 1.00x |
| 2 x 8  | 9.53 | 1.14x |
| 4 x 4  | **11.51** | **1.38x** |
| 8 x 2  | 4.50 | 0.54x — collapses |

So for seed sweeps and ablations, 4 processes of 4 threads beats one 16-thread job by
~40%. Do not push it further: at 8 processes throughput falls off a cliff (contention for
memory bandwidth across two NUMA sockets). For a single run that must finish soon, use a
large batch and all the threads instead.

### Measuring it

```bash
python -m lut_torch.bench_cpu                        # both experiments
python -m lut_torch.bench_cpu --experiment grid --threads 1,2,4,8,16 --batches 1,8,32
python -m lut_torch.bench_cpu --experiment concurrency --total-cores 32 --processes 1,2,4,8
```

`grid` sweeps threads x batch in one process and reports seq/s, speedup and parallel
efficiency. `concurrency` splits a fixed core budget `K` ways across independent processes
and reports aggregate throughput, which is the number that matters for a seed sweep. Run it
on the machine you will actually train on — a shared login node gives noisy numbers.

## Flags

`train.py --help` is authoritative. Notes on the ones with sharp edges:

- `--batch-size` — see above; this is the CPU-utilisation knob, and it also changes the
  gradient (it averages over the batch).
- `--optimizer {sgd,adam,adamw}` — `sgd` is the `main.py`-equivalent default.
  `--momentum` and `--weight-decay` affect dense parameters only.
- `--gradient-pair-selection {closest,cosine-distance}` — `cosine-distance` is meaningfully
  more expensive: its choice depends on the upstream gradient and on the table rows, so it
  varies per pair and per tree and must be assembled at expanded shape.
- `--new-MHA` is optional. Without it, each head emits the full `embedding_dim` and head
  outputs are summed into the residual with no `MHA_O` table, exactly as in `main.py`.
- `--load-model` resumes properly (model + optimizer + scheduler + step + RNG).
  `--reset-optimizer` loads weights only.
- `--lut-comparison-mode` accepts only `random`; it exists so `main.py` commands port over.
- `--device {cpu,cuda,cuda:N,auto}` — see "GPU" below. On GPU, batch size stops being a
  CPU-thread-efficiency knob and becomes the whole story: batch >= 32 is close to free
  there, batch 1 gets no benefit over CPU at all. `--kernels` is silently a no-op on GPU.

**Not carried over** (still available in `main.py`): old hard attention, `--soft-attention`,
`--factored-output`, `--fp16`, `no-overlap`/`overlap` comparison modes,
`--halve-lr-at-20k-40k`, `--save-multiple-checkpoints`, sampling/generation, and the
`LUTUpdateStats` instrumentation.

## Testing

```bash
.venv/bin/python test_torch_lut_primitive.py
.venv/bin/python test_torch_parity.py
.venv/bin/python test_torch_training.py
```

The venv has torch but no pytest; the cluster python has pytest but no torch. The tests
import `parametrize`/`run_all` from `_testing.py` so they run under either
(`python FILE` uses the shim runner, `python -m pytest FILE` uses real pytest). They import
`main.py` for comparison, so they need the venv.

| Suite | What it pins down |
|---|---|
| `test_torch_lut_primitive` | the primitive against `main.py`'s own kernels, plus `LUTFunction` vs `lut_lookup_reference` |
| `test_torch_parity` | whole-model step equality with `main.py` across 17 configurations |
| `test_torch_training` | batch equivalence, LR schedule vs `main.get_learning_rate`, checkpoint resume, evaluation |

### The parity contract

`test_torch_parity.py` builds a `main.Model` and a `LutModel` in one process from the same
seed and asserts, at batch 1 with SGD: identical construction, logits, loss, every gradient,
and identical tables after one optimizer step. Because `main.py` mutates `S` during backward,
its gradient is recovered as `(S_before - S_after) / lr`. A guard asserts every table carries
a non-zero gradient, so the comparison cannot pass vacuously.

**Any change to indexing, selection, or gradient routing must keep this green across the
whole config matrix.** If you add a structural flag, it must reach `LutConfig.validate`, the
`train.py` parser, the printed config, and `test_torch_parity.CONFIGS`.

## Kernels

`kernels.py` is a fused-Numba accelerator for `LUTFunction`, for `closest` selection on CPU
float32 only. `cosine-distance` always stays on the torch path in `functional.py` — its
selection depends on the upstream gradient, which the kernels do not implement.
`--kernels {auto,numba,torch}` (default `auto`) picks the backend; `set_kernel_backend` is a
module-level switch rather than a `lut_lookup` argument, deliberately, because adding an
argument to `LUTFunction.apply` would shift the `needs_input_grad` offset `backward` indexes
from (margins start at 7) — a change that fails silently as a wrong gradient, not a crash.

There are exactly two call shapes (`kernels._layout`): a single unpaired group (Q/K/V/FFN/
`MHA_O`/output) and the Final table's two groups (QKPE, unpaired and possibly tree-broadcast
under `--qkpe-direct-sign`; V-side, paired via `pair_index`). Each kernel's `prange` axis is
chosen so every iteration owns its output cells outright — see the docstring on each kernel
for the specific argument. The paired-group backward (`_k_grad_paired`) is the one worth
understanding: it parallelises over *source position*, not pair, using a CSR inverse of
`pair_index` (`NewAttentionHead.final_pair_ptr`/`final_pair_list`, built once at construction)
so that several pairs accumulating into one source position never race.

Every kernel has a genuinely serial mirror (`_k_*_serial`), generated at import time from the
parallel kernel's own source (`prange` → `range`, `parallel=True` dropped) rather than
hand-duplicated, so there is exactly one body per kernel to keep correct. `_pick` chooses
serial below `_SERIAL_BELOW_ITEMS` (4096 work items) and parallel, thread-capped by
`_threads_for`, above it — a parallel kernel capped to one thread was still measured slower
than a plain serial loop doing the same work (thread-pool dispatch has cost even at n=1).

**`fastmath` is never used.** It reorders float reductions, which risks the bit-exact parity
`test_torch_parity` exists to guarantee, and it measured no net win once the actual bottleneck
(dispatch, not arithmetic) was fixed.

**Honest performance note.** In isolation (a single Final-table call, tight loop, no
surrounding model) the kernels measured 1.8-3.7x faster than the torch path, and that is the
number the design work in this section was based on. Measured end-to-end, on this machine, at
the production shape, batch 1, 16 threads: **no consistent net win** (torch and kernel
backends land within a few percent of each other, run to run). Two things are true at once:
the kernels genuinely reduce per-call CPU time (confirmed by direct instrumentation — the
in-kernel time dropped as expected), and the *total* step time did not move, because at ~103
LUT calls/step the constant per-call cost around each kernel invocation (marshaling to numpy,
Python dispatch) is of the same order as what was saved, and this machine is a **shared,
non-exclusive, frequency-scaled node** (`CPU(s) scaling MHz: 47%` in `lscpu`, load average
observed 3-7 across unrelated jobs) whose measurement noise was, at one point, large enough to
produce a completely different conclusion between two consecutive runs of the identical
benchmark. Re-measure on an exclusive allocation before trusting either direction. The
straightforward next lever, if this doesn't pan out, is Track B (batch the heads of a layer):
it cuts the *call count* itself by `num_heads`x, which is the one thing every measurement
here agrees is the actual constraint — see "If you need more speed" below.

## Performance history

Head to head against `main.py` at the production shape, 16 threads, same host:

| | seq/s | vs `main.py` per sequence |
|---|---|---|
| `main.py` batch 1 (Numba) | 5.09 | 1.00x |
| `lut_torch` batch 1 | 3.35 | 0.66x |
| `lut_torch` batch 8 | 9.55 | 1.88x |
| `lut_torch` batch 32 | **12.21** | **2.40x** |

So the port is ~2.4x faster than the Numba path once batched, and still slower at batch 1 —
`main.py` processes one sequence per step by construction, so batching is where torch wins.
Do not compare at batch 1 and conclude the port is slow.

Five successive versions were 2-24x slower than this; if you touch the primitive,
re-measure, and know what caused those:

1. A mask-and-reduce `first_argmin` (two `amin` passes) was **62% of total runtime**.
2. Row bits and the closest-comparison index were computed on *pair-expanded* Final margins
   when they depend only on position — a factor of ~CS/2 of wasted work.
3. `closest` selection now assembles its whole margin gradient at source shape, allocating
   no `(B, n_pairs, N_T, R)` buffer. `cosine-distance` cannot do this.
4. `gi` is a batched mat-vec with the `S[j]` half hoisted out of the group loop, not an
   elementwise product plus reduction over a large temporary.
5. Unconditionally sparse table gradients — see "Table gradients". This one only showed up
   at batch 32 and cost 19x there, which is the argument for sweeping batch sizes when
   benchmarking rather than trusting a single point.
6. The fused kernels (see "Kernels" above), first cut: every call ran numba's parallel
   dispatch at the trainer's full configured thread count, which is fine for the Final table
   (~16000 visits) but actively harmful for the small tables (~1000 visits) — a parallel
   kernel capped to even one thread beats a genuinely serial kernel doing identical work, so
   thread count alone cannot fix small calls. Fixed with `_pick`/`_SERIAL_BELOW_ITEMS`. Net
   effect on this (shared, noisy) machine was still a wash end-to-end — see the honest note
   above — which is itself the lesson: an isolated tight-loop microbenchmark of one call does
   not establish what happens at ~103 calls/step interleaved with everything else the model
   does, and does not survive a non-exclusive machine at all. Re-measure on real allocation
   before acting on either number.

### If you need more speed

The model dispatches **~29,000 ATen ops per training step** (103 `lut_lookup` calls forward,
as many again backward, each tens of ops). That count is structural — it does not change
with batch size — and it is the binding constraint:

- On CPU it is why threads help so little at small batch: each op is tiny, so dispatch
  dominates and there is nothing to divide across threads.
- On GPU it sets a kernel-launch floor of roughly 90-230 ms/step regardless of batch, which
  is what a port would have to beat.

The two changes that would actually move it, in order: **batch the heads of a layer into
one set of ops** (cuts the op count ~`num_heads`x, helps CPU and GPU alike), then
`torch.compile` / CUDA graphs. A Numba kernel inside `LUTFunction.backward` remains
available as a last resort — CPU torch tensors are zero-copy `.numpy()`.

### GPU

Implemented and measured on an RTX 2080 Ti (`--device {cpu,cuda,cuda:N,auto}` in
`train.py`; `run_files/torch_POS_run_gpu.sh`). **No custom CUDA kernels** -- this is the
plain torch autograd path; `lut_lookup` already falls back off the fused-kernel backend on
any non-CPU tensor (`kernels.supported` checks `device.type`), so `--kernels` has no effect
on GPU and isn't in the GPU run script.

**Correctness first, since this matters more than speed:** built the same model on CPU and
GPU from the same seed, ran one step of SGD with `table_grad=dense`, and compared. Forward
logits were bit-exact; the loss and every parameter after the optimizer step agreed to
~1e-6/1e-10 (float32 rounding from summation order, not a bug -- CUDA's `index_add_`
accumulates with atomics, so exact reproducibility across runs on GPU isn't guaranteed the
way it is on CPU; the parity suite stays CPU-only and is unaffected). A 300-step run at the
production shape trained normally (perplexity 18.0 to 2.8, accuracy 5% to 66%), and
checkpoint save/resume round-trips correctly (RNG state now includes
`torch.cuda.get_rng_state_all()` alongside python/numpy/torch, so a resumed GPU run's
dropout stream is reproducible too).

**Performance -- this is the one number that actually matters**, production shape:

| batch | ms/step | seq/s |
|---|---|---|
| 1 | 172.4 | 5.80 |
| 8 | 173.1 | 46.21 |
| 32 | 177.6 | **180.20** |
| 128 | 435.9 | 293.63 |

Batch 1/8/32 cost almost exactly the same wall-clock time -- that is the ~30,000-ATen-op
kernel-launch floor from "If you need more speed" above, measured directly rather than
estimated: a step launches the same number of tiny ops regardless of batch size, and at
batch <= 32 those launches, not the arithmetic, are what the GPU spends its time on.
**At batch 1, GPU is no faster than CPU** (5.80 seq/s here vs 6.65-6.70 seq/s CPU torch/numba
at batch 1, 16 threads) -- both are dominated by the same kind of fixed per-step overhead,
paid on different hardware. Once batch size clears that floor, throughput is close to free:
batch 32 is **~15x the CPU torch path's own best batch-32 number** (180 vs 12.2 seq/s), with
zero kernel work. This was a deliberate choice, not an oversight: batching further to close
the batch-1 gap (custom `numba.cuda.jit` kernels mirroring `kernels.py`, mirrored per-call
race-freedom arguments and all) was evaluated and set aside in favour of this result -- the
effort-to-payoff ratio wasn't close if a large-enough batch is acceptable for the research
question at hand. Revisit that path specifically if genuine batch-1 GPU throughput is later
needed; the invariants and race-freedom design in `kernels.py`'s CPU kernels are the
template to port from.

**Setup:** `torch.cuda.is_available()` needs nothing beyond what's already in the venv
(torch's own bundled CUDA runtime, `cu130`). No `nvidia-smi` in `PATH` on the tested node --
`/dev/nvidia*` and `torch.cuda` are the reliable way to check. `numba.cuda.jit` (only
relevant if the custom-kernel path above is revisited) is a different story: it needs
`libnvvm.so` and `libdevice.10.bc`, which nothing in the venv ships. `pip install
nvidia-cuda-nvcc-cu12==12.4.131` (not the `cu13` variant -- installed and current at time of
writing) provides both, but two of numba 0.65.1's internal lookups need help finding them:
`get_supported_ccs()` needs an *unversioned* `libcudart.so` next to the versioned one from
torch's own CUDA runtime (`ln -s libcudart.so.13 libcudart.so` in
`nvidia/cu13/lib/`), and the CUDA_HOME-based nvvm lookup's regex requires a *versioned*
name where the cu12 pip package ships an unversioned one (`ln -s libnvvm.so libnvvm.so.4`
in `nvidia/cuda_nvcc/nvvm/lib64/`). With `CUDA_HOME=.../nvidia/cuda_nvcc` and
`LD_LIBRARY_PATH` including `nvidia/cu13/lib`, `from numba import cuda; cuda.is_available()`
and a trivial `@cuda.jit` kernel both work. None of this is needed for the plain-torch GPU
path above -- only for a future custom-kernel port.

## Rules for changes

- Keep `main.py` untouched. It is the reference the parity suite measures against.
- Parity first: get `test_torch_parity.py` green before trusting any result from a change.
- New structural flags must reach validation, parser, printed config, and the parity matrix.
- New tables must get a name, because instrumentation and analysis discover tables by name.
- Do not vectorise the RNG loops in `comparisons.py`.
- Do not switch the training objective to a mean without also rescaling the LR.
- If you add a table whose margins are shared across trees, keep the tree axis at size 1 and
  let it broadcast rather than expanding it.
- Preserve uncommitted work and research artifacts; never clean `results/`, checkpoints,
  caches, or W&B directories.

## Known gaps

- **Update statistics** are not implemented. The hook is in place: pass `capture={}` to
  `LutModel.forward` and every table records its row indices `j` and closest-comparison
  indices under the same names `LUTUpdateStats` uses (`ffn_0`, `head_0_0_Final`,
  `mha_o_0`, `output`). The `j` tensor is `(B, P, N_T)`; `r_closest` is per group at the
  margin source's own shape.
- **Inference** has no entry point here. `infer_lut_pos.py` reads `main.py`'s `.npz` files
  and cannot load these checkpoints.
- **Old attention** (hard and soft) is not ported. If it is ever wanted, it is the same
  primitive with three margin groups (Q, K, PE) instead of Final's two — `LutGroup` already
  covers the shape.
