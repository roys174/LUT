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
| `run_index.py` | Finds trained runs from `results/*/*/config.txt`. Stdlib only — no torch. |
| `inference.py` | Loads a checkpoint and samples from it. The generation half of the chat UI. |
| `chat_server.py` + `chat_ui/` | Web chat interface (`python -m lut_torch.chat_server`). |
| `_testing.py` | pytest shim so tests run on either interpreter here. |

Tests live at the repository root: `test_torch_lut_primitive.py`, `test_torch_parity.py`,
`test_torch_training.py`, `test_torch_chat.py`.

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

   That guarantee is only as good as the RNG state the builder is handed, and
   `test_torch_parity.py` builds both models back to back with nothing drawn in between,
   so it cannot see a divergence introduced *before* construction. `train.py` has exactly
   one draw there: `val_starts`. It uses Python `random.randint` in a loop, matching
   `main.load_training_data`'s `testing_input_data` (`main.py:4548`) draw for draw, so at
   the same seed both trainers hand `build_Model`/`LutModel` the same Python *and* NumPy
   streams and construct the same model. Switching it back to `np.random.randint` (as it
   was before) offsets the NumPy stream by `--testing-length` draws and every table in
   the model differs — with no error and no test failure, because the parity suite never
   goes through `train.py`. Anything new that draws before the model is constructed has
   to be mirrored in `main.py` or it breaks this the same silent way.

   Two streams still diverge after construction and are *not* aligned: training snippet
   starts (`main.py` draws with Python `random`, `train.py` with `np.random.randint`) and
   dropout masks (NumPy vs torch). Two runs of the same config therefore still take
   different paths; they just start from the same model and are scored on the same
   windows.

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
- `--output-code-dim D` / `--output-code {learned,binary}` — the output table emits `D`
  numbers instead of one score per class, and a `(V, D)` code matrix `C` maps them to the
  logits, so the table costs `n_t * 2**n_c * D` entries rather than `n_t * 2**n_c * V`. At
  the default shape and POS's 18 classes that is 73,728 → 20,480 floats, which is not worth
  having on its own; the flag exists because the same head at `V = 14,398` is 59M → 57k, and
  POS is where the accuracy cost of the change can be measured cheaply.
  **Settled value at `V = 14,398`: `--output-code-dim 112 --output-code learned
  --output-code-bias`** — 133.4 perplexity from a 3,461,982-parameter head, 68x smaller than
  the dense 235,896,832-entry one. `learned` saturates there (`D = 224/448/1000` are within
  1.1%, and their train/val gap grows monotonically, so the extra width only memorises);
  `D = 56` costs 1.2% if the head must be smaller still. See DECISIONS.md Part II.
  Three things to know. **The logits are `y @ C.T` and that is not a relaxation**: scoring
  class `k` by treating the `D` coordinates as independent bit comparisons,
  `sum_b [c_kb log sigma(y_b) + (1-c_kb) log sigma(-y_b)]`, differs from `sum_b c_kb y_b`
  only by a term independent of `k`, which the softmax cancels — so the "compare each
  coordinate against zero" head *is* a linear readout, with no thresholding in the loss and
  no surrogate needed for it (`test_torch_output_code.py`). **Use `learned`; `binary` and
  `random` are controls, not options.** Measured at `V = 14,398` (DECISIONS.md, Decision L3),
  learned `D = 14` beats frozen binary `D = 1000` — a >70x width disadvantage with no
  crossover at any `D` tested. Widening `D` does not rescue a frozen code: frozen random at
  `D = 28` scores the same as frozen binary at `D = 14` despite a far larger minimum
  distance, so it is the freezing that costs, not the code's Hamming distance. (`binary`'s
  minimum distance is in any case stuck at 2 for every `D` from 28 to 1000 — a defect in
  `ecoc_masks`' greedy construction, not a property of the vocabulary. Fixing it is not
  expected to change the ranking.) **It caps the logit matrix at rank `D`** — for an argmax
  metric like POS accuracy that may cost nothing, but perplexity feels it directly, so
  report validation loss alongside accuracy or the result will not extrapolate to full
  vocabulary.
- `--output-code-bias` — adds a per-class bias, `logits = y @ C.T + b`. Without it every
  logit vector lies in `col(C)`, class priors included, and a frozen code cannot choose a
  column space that accommodates them: `col(C_binary)` holds 20.8% of this corpus's
  log-prior energy, less than a random D-dim subspace (27.9%). That is the leading
  explanation for `binary`'s loss nearly doubling while its accuracy fell under a point —
  ranking well but calibrating badly is what a missing prior looks like. Costs `V`
  parameters. Requires `--output-code-dim`.
- `--output-code-lr-scale` — multiplies the learning rate of `C` and `b` only. A token's
  `y` is the sum of `n_t` table rows and every one receives the full `dL/dy`, so a step
  moves the table's contribution to the output by roughly `n_t x` nominal while `C` and `b`
  move by `1x`. At POS that asymmetry is cheap to ignore (`C` is 0.1% of the head); at a
  full vocabulary `C` is about half of it (`V*D = 201,572` against a `16,384*14` table), so
  the rule has to be right before the expensive run, and POS is the only place a
  `y_dim = V` baseline exists to measure it against.
  The flag has no `main.py` counterpart and so cannot enter `test_torch_parity.CONFIGS`;
  what protects parity is that the code matrix is drawn last and only when the flag is set,
  leaving every existing configuration bit-identical (`test_default_head_unchanged`).
- `--lm-validate-last-position-only` — LM validation scores every position by default
  now, not just the last. `main.py`'s `StandardOutputHead.validation_loss` scores only the
  last, and this flag restores that. The forward computes all `CS` positions either way, so
  the old default discarded 31 of 32 predictions and left `--testing-length` scored tokens
  in total. **The two modes are not comparable**: the last position is the only one with a
  full context, so its loss is systematically lower. Any LM number recorded before this
  change was measured the old way.
- `--best-checkpoint-metric {auto,acc,loss}` — `auto` selects on accuracy for POS and on
  validation loss for LM. Perplexity is the LM metric and loss is the more sensitive
  signal; it also improves far less often than accuracy early in a run, which matters when
  a full-vocabulary checkpoint is ~1.1 GB and the best copy is written on every
  improvement.
- **`<unk>` in LM validation.** `<unk>` is 6.9% of this corpus and the LM branch trains on
  it as an ordinary target (POS masks targets whose *input* was `<unk>`, `data.py:88`;
  LM does not). `evaluate` therefore returns both scorings — `loss`/`acc` over every
  target, and `loss_no_unk`/`acc_no_unk` over the same targets with `<unk>` removed — and
  both reach `loss.csv` and W&B. Training is untouched, so nothing is committed to. Use
  the no-unk pair to compare runs; a headline perplexity is not comparable across the two.
- `--lut-comparison-mode` accepts only `random`; it exists so `main.py` commands port over.
- `--device {cpu,cuda,cuda:N,auto}` — see "GPU" below. `--kernels` is silently a no-op on
  GPU.
- `--stack-heads {auto,on,off}` — run a layer's heads as 4 `lut_lookup` calls instead of
  `4 * H` (`StackedHeads`). `auto` is on whenever `num_heads > 1`. 3.8x on GPU and 2.0x on
  CPU at the production shape. **It changes the parameter layout**, so the state dict keys
  differ; `--load-model` converts a per-head checkpoint automatically (`stacking.py`) and
  says so, discarding optimizer state because the parameter grouping changed. The forward
  is bit-identical to the unstacked path and single-layer gradients are too; deeper
  gradients differ by 1-3 float32 ULP because fusing `H` per-head gathers reassociates the
  scatter-add into the layer input. That is why the unstacked path still exists and why
  `test_torch_parity.py` builds models unstacked.
- `--epochs` — budget the run in passes over the training tokens; `max_steps` is derived
  per batch size in `derive_schedule` (`ceil(epochs * positions / (batch * CS))`). A step
  count is not comparable across batch sizes: 200k steps is 3 passes over the POS corpus
  at batch 1 and 96 at batch 32, and because the objective is a batch *mean*, 32x the
  total parameter travel to go with it. Any batch-size comparison budgeted in steps is
  measuring those two things instead.
- `--lr-batch-scaling {none,linear,sqrt}` — the other half of `--epochs`. At a fixed epoch
  budget, batch B takes 1/B as many steps of the same size, so leaving the LR alone turns
  "same epochs" into "B times less optimisation" — the mirror image of the confound
  `--epochs` removes. Total drift and the gradient-noise scale both go as `lr/B`, so
  `linear` matches both at once. `--lr-reference-batch-size` (default 1) is the batch size
  `--max-learning-rate` is quoted at, so batch-1 runs are unaffected by default.
- `--lr-decay-start` (t0) — the step at which inverse-sqrt decay begins; the LR holds flat
  at the peak from the end of warmup until here, so the schedule is ramp → hold → decay →
  cooldown. It exists because `(W / s) ** alpha` puts the warmup length in the *decay* as
  well as the ramp: lengthening the warmup raises the LR at every later step, which made
  every warmup comparison in the LR probes uninterpretable (a short ramp "won" twice, both
  times by being less exposed to a too-high LR). Defaults to the warmup length, where the
  `min(1, ...)` never binds and the curve is bit-identical to what this project has always
  run. Must be ≥ the resolved warmup — starting the decay inside the ramp would step the LR
  down discontinuously, and `derive_schedule` refuses it. The depth reached before cooldown
  is `(t0 / max_steps) ** alpha`, reported as `lr_decay_depth`.
- `--lr-cooldown-frac` — linear ramp to `--lr-cooldown-min-ratio` over the tail of the run,
  on top of whichever scheduler is chosen. `inverse-sqrt` never anneals: it approaches
  `(warmup / max_steps) ** alpha` and stops, leaving real LR on for however long the run
  continues past its optimum. At 0 the schedule is bit-identical to `main.get_learning_rate`
  and `test_schedule_matches_main` covers that.
- `--warmup-frac` / `--warmup-min-steps` — warmup as a fraction of `max_steps` rather than a
  count. Under an `--epochs` budget the fraction holds warmup to the same number of *tokens*
  at every batch size and the schedule to the same end depth (`frac ** alpha`); a fixed
  count does neither once `max_steps` varies. `--warmup-min-steps` deliberately breaks the
  depth match on short runs, where the fair fraction is a few dozen steps. The realised
  depth is recorded per run as `lr_decay_depth`.
- `--validation-interval-epochs` / `--wandb-log-points` — cadence measured against data
  instead of the step counter, so curves from different batch sizes land on the same epoch
  values and carry the same number of points. Without this, overlaying a batch-1 and a
  batch-32 run compares points taken 32x apart in data, and W&B's point-counted smoothing
  window smooths them by different amounts.
- `--save-model` also writes `..._best.pt` at the best validation accuracy so far, and the
  run's summary carries `val/best_acc` / `val/best_step` / `val/best_epoch`. The final
  checkpoint is not the best one — under a fixed-step schedule these runs have peaked as
  early as 18% in.
- `--cuda-graphs {auto,on,off}` — replay forward+backward from a single launch
  (`graphs.py`). `auto` is on for `--device cuda`, a no-op on CPU. Numerically identical.
  Forces `--table-grad dense`; refuses to combine with an explicit `--table-grad sparse`.
  The optimizer and LR scheduler stay eager on purpose — see the module docstring.

**Not carried over** (still available in `main.py`): old hard attention, `--soft-attention`,
`--factored-output`, `--fp16`, `no-overlap`/`overlap` comparison modes,
`--halve-lr-at-20k-40k`, `--save-multiple-checkpoints`, sampling/generation, and the
`LUTUpdateStats` instrumentation.

## Training recipe

```
--epochs 10                         # or 8 with --lr-cooldown-frac 0.25
--lr-batch-scaling linear --lr-reference-batch-size 1
--max-learning-rate 0.0025          # 0.00125 if budgeting 3 epochs
--warmup-steps $(( 6400 / BATCH ))
--lr-cooldown-frac 0.2
--validation-interval-epochs 0.1
--wandb-log-points 3000
--checkpoint-interval-validations 10
```

Budget in epochs, never in steps -- `--max-steps` means 3 passes over this corpus at batch
1 and 96 at batch 32. Scale the peak LR linearly with batch. Set warmup to `6400/batch`,
which holds both warmup tokens and decay depth constant across batch sizes. Always keep a
terminal cooldown, but do not tune its shape: having one is worth a great deal and its
length, start and floor are all measurably irrelevant.

Batch size is a speed knob, not an accuracy knob. Prefer `heads 4` over 8 -- 1.71x faster
for no measurable gain.

Before comparing two runs, know the noise floors: **0.0005** accuracy (0.0030 val loss)
between jobs at an identical config and seed, and **0.0012-0.0031** between seeds, because
`--seed` also draws the validation windows. Compare configurations within a seed. Val loss
is the more sensitive metric; train loss says nothing useful about budget.

Every number behind these choices, the experiments that produced them, the wrong turns,
and what is still open are in `DECISIONS.md` at the repo root. Change these defaults only
against that file.

## Talking to a trained model

```bash
srun --gres=gg:g0:1 --mem=16g -c4 --time=2:00:00 --pty run_files/chat_server.sh
python -m lut_torch.chat_server --device cpu    # or wherever the venv works
```

`srun` and not `sbatch` — unlike everything else in `run_files/`, this is a service you
sit in front of rather than a run you collect afterwards, so the tunnel line belongs on
your terminal and Ctrl-C should stop it. `sbatch` is still there for leaving it up
overnight, at the price of an allocation held for the full `--time` whether or not
anyone is using it.

A browser chat over the LM checkpoints. Stdlib `http.server` plus three static files
under `chat_ui/` — nothing to install, which is the point on a node with no internet.

Three things about the model decide the design, and all three are easy to get wrong:

1. **It is a next-token model, not an assistant.** A "conversation" is one growing piece
   of text: the UI keeps the transcript verbatim, sends it as the prompt, and the reply
   is the continuation. Nothing in the corpus taught it turns.

2. **The context is fixed and there is no KV cache.** A prompt shorter than
   `context_size` goes at positions `0..n-1` and the prediction is read at `n-1`; longer,
   and the window slides. Trailing padding is invisible because attention is causal, and
   `test_short_prompt_ignores_padding` checks that against the model rather than trusting
   it. Left-padding would be wrong twice: the PE is absolute, and position 0 receives no
   attention output at all. Each generated token costs a full forward over the window.

3. **`model.eval()` is mandatory.** `LutLayer._residual_dropout` is live in `train()`
   mode, so a forgotten `eval()` samples every reply from a different perturbed model —
   and nothing raises.

Listing the runs never opens a checkpoint. `run_index.py` reads `config.txt` and
`loss.csv` only, and classifies a run by **the recorded command** — LM is
`lut_torch.train` without `--POS-task` — because sweep directory names drift and
`results/GPT_POS` is not this trainer at all. Opening a 1.1 GB baseline checkpoint per
dropdown entry is the thing that rule exists to avoid.

A live run rewrites `model_best.pt` on every validation improvement, so loading can lose
that race; `load_checkpoint_file` retries and the server falls back to the periodic copy.

`<unk>` is suppressed by default during sampling. It is 6.9% of wiki2 and its single most
frequent token, so unsuppressed sampling emits little else. The input tokenizer view is
there for the same reason from the other direction: the vocabulary keeps only types seen
10+ times, and a reply that ignores half the prompt usually has that as its cause.

## Testing

```bash
.venv/bin/python test_torch_lut_primitive.py
.venv/bin/python test_torch_parity.py
.venv/bin/python test_torch_training.py
.venv/bin/python test_torch_kernels.py
.venv/bin/python test_torch_stacking.py
.venv/bin/python test_torch_graphs.py     # skips itself without a GPU
.venv/bin/python test_torch_chat.py       # its torch-free half runs anywhere
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
| `test_torch_kernels` | the fused Numba backend against the torch path, both layouts |
| `test_torch_stacking` | stacked heads vs unstacked: exact forward, exact single-layer gradients, float32-epsilon bound deeper, call-count reduction, checkpoint conversion |
| `test_torch_graphs` | CUDA-graph replay vs eager over *consecutive* steps, dropout redraw, and the zero_grad misuse guard. Skips itself without a GPU |
| `test_torch_chat` | run discovery and LM/POS classification, incremental decoding against `WordTokenizer.decode`, sampling filters, the causal-padding assumption, and the server's routes against a stub. The run-index and HTTP halves need no torch |

`test_torch_stacking` is the reason stacking is safe to trust: parity pins unstacked
against `main.py`, and stacking pins stacked against unstacked, so the guarantee carries
across without rewriting the parity suite to understand a second parameter layout.

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

Both of the changes that move it are now implemented, and both attack the launch count
rather than the arithmetic — see "GPU" below for the numbers:

- **`--stack-heads`** runs a layer's heads as 4 `lut_lookup` calls instead of `4 * H`
  (`StackedHeads` in `modules.py`). This is the one that also helps CPU.
- **`--cuda-graphs`** replays the whole forward+backward from one launch (`graphs.py`).
  GPU only, by construction.

What is left after those is genuinely fused kernels; `lut_torch/PLAN_gpu_kernels.md` has
the measured case for and against, and concludes with Triton rather than `numba.cuda`.

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

Re-checked with `--stack-heads` and `--cuda-graphs` both on: a 1,500-step run at the
production shape trained normally (perplexity 18.0 → 6.35 → 6.18, accuracy 5% → 52%),
saved, and resumed cleanly — the graph is captured lazily on the first step *after* any
checkpoint load, so resume re-captures against the restored weights rather than replaying
a stale one (perplexity 3.39, accuracy 59% at the post-resume validation).

**Where the time went, before either optimisation** (profiler, production shape, batch 1):

```
wall                         373 ms/step
GPU busy                      94 ms/step      <-- 25% of wall
CUDA kernel launches/step  29,214
aten:: calls/step          59,029
```

Not compute-bound; **launch-bound twice over**. ~280 ms is Python + autograd + ATen
dispatch + `cudaLaunchKernel` at ~12 us a launch, and the 94 ms of "GPU busy" is itself
29,214 kernels at this card's ~3.2 us minimum kernel duration -- almost none of it
arithmetic. Two scaling curves confirm it: cost was linear in head count (1/2/4/8 heads →
63/107/182/370 ms) and flat in batch size (1/4/16 → 380/342/430 ms). Sixteen times the
work for 13% more time.

There is no missing parallelism *inside* a sample -- every `(position, tree)` is already
an independent thread. There is just not enough of it at batch 1. Against this card's
69,632 resident threads: a Q/K/V/FFN call is 1,024 work items (**1.5%** of the machine), a
Final call 15,872 (22.8%). The fix is to make independent *tables* share a launch, which
is what `--stack-heads` does (8 heads stacked: 8,192 and 126,976 items).

**Measured end to end on the real trainer**, production shape, batch 1, 400 steps,
steady-state rate:

| | step/s | ms/step | speedup |
|---|---|---|---|
| neither | 2.92 | 342 | 1.0x |
| `--stack-heads` only | 15.94 | 62.7 | 5.5x |
| `--cuda-graphs` only | 10.04 | 99.6 | 3.4x |
| **both** | **24.32** | **41.1** | **8.3x** |

They are **not** multiplicative (5.5 x 3.4 would be 18.7x), because they overlap: both
attack launch cost, just at different ends. Stacking removes ~8/9 of the launches; graphs
remove the CPU price of whatever launches remain, so there is less left for them to remove
once stacking has run. Adding graphs on top of stacking is a further 1.53x; adding stacking
on top of graphs is a further 2.42x. Take both. Loss curves are identical across all four.

Batch size still buys sequences/second but is no longer the only lever it was. For
reference, the pre-optimisation batch scan was 172/173/178/436 ms/step at batch 1/8/32/128
-- i.e. batch 32 was the only way to get value out of the card. It no longer is.

**What is left, and would a bigger card help?** Under a graph the step breaks down as:

| | forward | backward | optimizer | launches | launch floor |
|---|---|---|---|---|---|
| unstacked | 23.3 | 77.0 | 12.9 | 29,820 | 95 ms (**84%** of the step) |
| stacked | 4.5 | 38.2 | 12.3 | 4,106 | 13 ms (**24%**) |

(Measured with gradients zeroed inside the graph, which is why the totals are above the
trainer's — the trainer uses the cheaper assign-on-first-backward path. The split is what
matters here.)

So stacking moved the model *out* of the launch-floor regime: 76% of a stacked step is now
real memory-bound work, where before 84% was launch overhead. Two consequences:

- **A faster card now helps, where before it barely would.** The bandwidth-bound 76%
  scales with memory bandwidth (this card: 616 GB/s, and the optimizer alone moves ~3.7 GB
  per step at ~300 GB/s effective). Scaling that part and halving the launch floor puts an
  A100 at roughly 2-2.5x and an H100 at roughly 3-3.5x this card. That is an estimate from
  the composition above, not a measurement — no second GPU was available to check it.
  A bigger card also lifts the batch ceiling a long way (peak memory here is 1.94 GB of
  11.3 GB).
- **Fused kernels are still worth more than a bigger card**, and would make a bigger card
  pay off better, since they cut both the remaining 4,106 launches and the intermediate
  memory traffic. Backward is 3x forward and is the target.
  `lut_torch/PLAN_gpu_kernels.md` argues for Triton (already installed, graph-capturable,
  needs no `nvcc` — there is none on the node) over `numba.cuda.jit`. Its 10-18 ms estimate
  predates stacking and should be re-derived before anyone commits to that work.

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
