# PERFORMANCE.md — `lut_torch/` runtime and memory

Measurements and the reasoning behind them for the torch port's speed and memory. The
rules that follow from this are summarised in `AGENTS.md` ("Performance in brief"); this
file keeps the evidence, so a later speed-up attempt starts from what was already tried.

If you touch the primitive, re-measure. Several earlier versions were 2-24x slower than the
current one without looking like bugs (see "Performance history").

## Head to head against `main.py`

Production shape (context 32, embedding 64, 6 layers, 4 heads, `n_t=64`, `n_c=6`, weight
sharing 8, frozen PE, direct-sign QKPE), 16 threads, same host:

| | seq/s | vs `main.py` per sequence |
|---|---|---|
| `main.py` batch 1 (Numba) | 5.09 | 1.00x |
| `lut_torch` batch 1 | 3.35 | 0.66x |
| `lut_torch` batch 8 | 9.55 | 1.88x |
| `lut_torch` batch 32 | **12.21** | **2.40x** |

The port is ~2.4x faster than the Numba path once batched, and still slower at batch 1 —
`main.py` processes one sequence per step by construction, so batching is where torch wins.
Do not compare at batch 1 and conclude the port is slow.

## Table gradients: sparse vs dense

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

`--table-grad auto` picks dense once visits reach the table's row count. Measured head to
head on the Final table, dense wins at every batch size including batch 1 (1.8x at B=1,
3.4x at B=32).

## CPU

`main.py` had four parallelism knobs: `--num-workers` (thread pool over batch elements),
`--replay-workers` (thread pool over independent `S` tables during sparse replay),
`--head-workers` (thread pool over MHA heads at batch 1), and `--numba-threads` (`prange`
inside each kernel). **All four are gone.**

`lut_torch` has exactly one: **ATen's intra-op thread pool**, set by
`torch.set_num_threads(args.cpu_cores)`. Each individual tensor operation splits its own
work across that pool. There is no parallelism *between* operations: the Python-level loop
over `num_layers x num_heads` runs one op at a time.

Consequences:

- **Batch size, not thread count, is what makes cores usable.** At batch 1 the per-head
  tensors are small and dispatch overhead dominates, so extra threads buy very little.
  Raising the batch multiplies the work per op while the overhead stays fixed. Threads and
  batch have to be tuned together — that is what `bench_cpu.py --experiment grid` measures.
- **The serial spine is Python dispatch.** Roughly `num_layers x num_heads` head calls per
  step, each a sequence of ops. Threads do not reduce that count, which is why per-sequence
  throughput at batch 1 does not approach the Numba path no matter how many cores you give it.
- **Never oversubscribe.** Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to the same value as
  `--cpu-cores` (`torch_POS_run.sh` does this).
- **Physical cores, not hyperthreads, and one socket if you can.** The big table gathers
  are memory-bound, so threads past the physical core count mostly add contention, and
  spilling across NUMA sockets adds latency. `torch.set_num_interop_threads` is irrelevant —
  it applies only to explicitly forked autograd subgraphs, which this model never creates.
- **Several narrow jobs beat one wide one.** Intra-op scaling is sublinear, so `K`
  processes of `T` threads generally deliver more total throughput than one process of
  `K*T` threads.

Memory scales with batch: roughly 0.9 GB at batch 1 and 2.2 GB at batch 16 at the POS
production shape. Size `--mem` accordingly.

### Measured scaling

Production shape without MHA, 20 physical cores, shared host — indicative, not exact.

| batch | 1 thr | 2 thr | 4 thr | 8 thr | 16 thr |
|---|---|---|---|---|---|
| 1  | 1.80 | 2.66 | 3.29 | 3.39 | 3.37 seq/s |
| 8  | 2.13 | 3.58 | 5.69 | 8.05 | 8.05 seq/s |
| 32 | 1.66 | 2.92 | 5.78 | 9.20 | **11.94 seq/s** |

Parallel efficiency at 4 threads: 46% at batch 1, 67% at batch 8, 87% at batch 32. At
batch 1 the curve is flat past 4 threads; at batch 32 it is still climbing at 16.

Splitting a fixed 16-core budget across independent processes (batch 8 each):

| processes x threads | aggregate seq/s | vs one wide job |
|---|---|---|
| 1 x 16 | 8.34 | 1.00x |
| 2 x 8  | 9.53 | 1.14x |
| 4 x 4  | **11.51** | **1.38x** |
| 8 x 2  | 4.50 | 0.54x — collapses |

At 8 processes throughput falls off a cliff (memory-bandwidth contention across two NUMA
sockets). For a single run that must finish soon, use a large batch and all the threads.

### Measuring it

```bash
python -m lut_torch.bench_cpu                        # both experiments
python -m lut_torch.bench_cpu --experiment grid --threads 1,2,4,8,16 --batches 1,8,32
python -m lut_torch.bench_cpu --experiment concurrency --total-cores 32 --processes 1,2,4,8
```

`grid` sweeps threads x batch in one process and reports seq/s, speedup and parallel
efficiency. `concurrency` splits a fixed core budget `K` ways across independent processes
and reports aggregate throughput. Run it on the machine you will actually train on — a
shared login node gives noisy numbers.

## Fused Numba kernels (CPU)

The design of `kernels.py` is in `AGENTS.md` ("Kernels"). What it measured:

In isolation (a single Final-table call, tight loop, no surrounding model) the kernels
measured 1.8-3.7x faster than the torch path. End-to-end, at the production shape, batch 1,
16 threads: **no consistent net win** — the two backends land within a few percent of each
other, run to run. Both are true at once: the kernels genuinely reduce per-call CPU time
(confirmed by instrumentation), and total step time did not move, because at ~103 LUT
calls/step the constant per-call cost around each invocation (marshaling to numpy, Python
dispatch) is of the same order as what was saved.

The measuring machine was a **shared, non-exclusive, frequency-scaled node**
(`CPU(s) scaling MHz: 47%`, load average 3-7 from unrelated jobs), noisy enough to produce
opposite conclusions on two consecutive runs of the identical benchmark. Re-measure on an
exclusive allocation before trusting either direction. The lesson generalises: an isolated
microbenchmark of one call does not establish what happens at ~103 calls/step interleaved
with everything else.

## GPU

Measured on an RTX 2080 Ti (`--device cuda`; `run_files/torch_POS_run_gpu.sh`). **No custom
CUDA kernels** — this is the plain torch autograd path; `lut_lookup` falls back off the
fused-kernel backend on any non-CPU tensor, so `--kernels` has no effect on GPU.

**Correctness.** Same model on CPU and GPU from the same seed, one SGD step with
`table_grad=dense`: forward logits bit-exact; loss and every parameter after the step agree
to ~1e-6/1e-10 (float32 summation order — CUDA's `index_add_` uses atomics, so GPU runs are
not bit-reproducible across runs; the parity suite stays CPU-only). Short training runs and
checkpoint save/resume were checked with and without `--stack-heads`/`--cuda-graphs`; the
graph is captured lazily on the first step *after* a checkpoint load, so resume re-captures
against the restored weights. RNG state includes `torch.cuda.get_rng_state_all()`.

### Where the time went (before stacking and graphs)

Profiler, production shape, batch 1:

```
wall                         373 ms/step
GPU busy                      94 ms/step      <-- 25% of wall
CUDA kernel launches/step  29,214
aten:: calls/step          59,029
```

Not compute-bound; **launch-bound twice over**. ~280 ms is Python + autograd + ATen
dispatch + `cudaLaunchKernel` at ~12 us a launch, and the 94 ms of "GPU busy" is itself
29,214 kernels at this card's ~3.2 us minimum kernel duration — almost none of it
arithmetic. Cost was linear in head count (1/2/4/8 heads → 63/107/182/370 ms) and flat in
batch size (1/4/16 → 380/342/430 ms): sixteen times the work for 13% more time.

There is no missing parallelism *inside* a sample — every `(position, tree)` is already an
independent thread; there is just not enough of it at batch 1. Against this card's 69,632
resident threads, a Q/K/V/FFN call is 1,024 work items (1.5%), a Final call 15,872 (22.8%).
The fix is to make independent *tables* share a launch, which is what `--stack-heads` does
(8 heads stacked: 8,192 and 126,976 items).

### Stacking and CUDA graphs

End to end on the real trainer, production shape, batch 1, 400 steps, steady state:

| | step/s | ms/step | speedup |
|---|---|---|---|
| neither | 2.92 | 342 | 1.0x |
| `--stack-heads` only | 15.94 | 62.7 | 5.5x |
| `--cuda-graphs` only | 10.04 | 99.6 | 3.4x |
| **both** | **24.32** | **41.1** | **8.3x** |

Not multiplicative (5.5 x 3.4 would be 18.7x) because both attack launch cost from
different ends: stacking removes ~8/9 of the launches; graphs remove the CPU price of
whatever launches remain. Graphs on top of stacking is a further 1.53x; stacking on top of
graphs a further 2.42x. Take both. Loss curves are identical across all four. (Stacking
also gives 2.0x on CPU.)

The pre-optimisation batch scan was 172/173/178/436 ms/step at batch 1/8/32/128 — batch 32
was then the only way to get value out of the card. It no longer is.

### Training throughput by batch size

Real POS training runs (`results/POS_torch_expC/`, 3 epochs, token-matched schedule, both
optimisations on; DECISIONS.md D6):

| batch | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| throughput | ~3,000 tok/s | 6,400 | 6,050 | 7,000 | 5,900 |

Throughput saturates at batch 8 and is flat to 128; only batch 1 is penalised, at roughly
half.

### What is left

Under a graph the step breaks down as (ms; measured with gradients zeroed inside the graph,
so totals are above the trainer's — the split is what matters):

| | forward | backward | optimizer | launches | launch floor |
|---|---|---|---|---|---|
| unstacked | 23.3 | 77.0 | 12.9 | 29,820 | 95 ms (**84%** of the step) |
| stacked | 4.5 | 38.2 | 12.3 | 4,106 | 13 ms (**24%**) |

Stacking moved the model *out* of the launch-floor regime: 76% of a stacked step is now
real memory-bound work. Two consequences:

- **A faster card now helps**, where before it barely would. The optimizer alone moves
  ~3.7 GB per step at ~300 GB/s effective on this 616 GB/s card. Estimated (not measured —
  no second GPU was available): A100 ~2-2.5x, H100 ~3-3.5x. Peak memory was 1.94 GB of
  11.3 GB, so a bigger card also lifts the batch ceiling a long way.
- **Fused kernels are still worth more than a bigger card**, and would make one pay off
  better, since they cut both the remaining 4,106 launches and the intermediate memory
  traffic. Backward is 3x forward and is the target. Triton was preferred over
  `numba.cuda.jit` (already installed, graph-capturable, needs no `nvcc`, which the nodes
  lack). An earlier 10-18 ms/step estimate for that work predates stacking and must be
  re-derived. (The planning note for it, `PLAN_gpu_kernels.md`, is no longer in the tree.)

## Performance history — what made earlier versions slow

Each of these cost 2-24x and none looked like a bug:

1. **`first_argmin` by mask-and-reduce** (two `amin` passes) was 62% of total runtime.
   Now `torch.min(dim=-1).indices`, which also gives the required first-minimum tie-break.
2. **Pair-expanded Final work.** Row bits and the closest-comparison index were computed on
   pair-expanded Final margins when they depend only on position — ~CS/2 wasted work.
3. **Expanded margin-gradient buffers.** `closest` selection now assembles its margin
   gradient at source shape with no `(B, n_pairs, N_T, R)` buffer. `cosine-distance` cannot.
4. **`gi` as elementwise product + reduction** over a large temporary; now a batched
   mat-vec with the `S[j]` half hoisted out of the group loop.
5. **Unconditionally sparse table gradients** (see above). Only showed up at batch 32, where
   it cost 19x — the argument for sweeping batch sizes when benchmarking.
6. **Fused kernels at full thread count.** Every call ran numba's parallel dispatch at the
   trainer's thread count: fine for Final (~16,000 visits), harmful for small tables
   (~1,000), and a parallel kernel capped to one thread is still slower than a serial one.
   Fixed with `_pick`/`_SERIAL_BELOW_ITEMS`; net end-to-end effect still a wash (see above).

## If you need more speed

The model dispatches **~29,000 ATen ops per training step** unstacked (103 `lut_lookup`
calls forward, as many backward, each tens of ops). That count is structural — it does not
change with batch size — and was the binding constraint on both CPU (each op is tiny, so
threads have nothing to divide) and GPU (a launch floor of ~90-230 ms/step). `--stack-heads`
and `--cuda-graphs` both attack that count and are implemented. What is left is genuinely
fused kernels (Triton) for the backward.

## Appendix: `numba.cuda` setup

Not needed for the plain-torch GPU path — only if a custom-kernel port via `numba.cuda` is
revisited. `torch.cuda` needs nothing beyond the venv (torch's bundled `cu130` runtime).
`numba.cuda.jit` needs `libnvvm.so` and `libdevice.10.bc`, which the venv does not ship.
`pip install nvidia-cuda-nvcc-cu12==12.4.131` (not the `cu13` variant) provides both, but
numba 0.65.1 then needs two symlinks:

- `ln -s libcudart.so.13 libcudart.so` in `nvidia/cu13/lib/` (`get_supported_ccs()` wants
  an unversioned name);
- `ln -s libnvvm.so libnvvm.so.4` in `nvidia/cuda_nvcc/nvvm/lib64/` (the CUDA_HOME nvvm
  lookup wants a versioned one).

With `CUDA_HOME=.../nvidia/cuda_nvcc` and `LD_LIBRARY_PATH` including `nvidia/cu13/lib`,
`cuda.is_available()` and a trivial `@cuda.jit` kernel both work.
