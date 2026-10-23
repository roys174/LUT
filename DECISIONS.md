# DECISIONS.md

Why the training configuration is what it is. `AGENTS.md` and `lut_torch/AGENTS.md` carry
the verdict; this file carries the measurements behind it, so that a later change can see
which numbers it would be overturning rather than rediscovering them. How these were
reached — the sweeps in order, the wrong turns, the conclusions later retracted — is in
`HISTORY.md`.

Part I is the POS task, Part II the LM coded output head on WikiText-2, Part III the move
to WikiText-103. Part I and Part II conclusions do not transfer in either direction --
where they conflict (anneal shape, budget), the regimes differ and both are correct for
their own task.

Part I measured on the POS task (wiki2, 2,156,877 training positions, context 32) at
`heads 4 / new-MHA on / weight-sharing 8`, batch 32, unless stated otherwise.

---

## The settled configuration

```
--epochs 10                         # or 8 with --lr-cooldown-frac 0.25
--lr-batch-scaling linear --lr-reference-batch-size 1
--max-learning-rate 0.0025          # 0.00125 if budgeting 3 epochs -- see Decision 3
--warmup-steps $(( 6400 / BATCH ))
--lr-cooldown-frac 0.2
--validation-interval-epochs 0.1
--wandb-log-points 3000
--checkpoint-interval-validations 10
```

Best single configuration measured: `batch 32 / h4 / ws8 / new-MHA on` at lr 0.08,
48M table entries, **0.9454 accuracy / 0.1583 val loss in ~57 min**. The pre-existing
schedule reached 0.9445 at 3.30 h on the same architecture and *ended* at 0.9407 after
9.02 h.

---

## Noise floors -- read this before comparing anything

Three different floors, each measured, each larger than it looks:

| what varies | accuracy | val loss | measured from |
|---|---|---|---|
| nothing (same config, same seed, same job repeated) | 0.0004 | -- | 3 repeats of bs32/lr0.04/E3 |
| same config and seed, **different job** | 0.0005 | **0.0030** | 3 replicate groups across expA/expB/probe2/sweep |
| **different seed**, same config | 0.0012-0.0031 | up to 0.0080 | expE, 5 configs x 2 seeds |

Two consequences that have already caught us out:

- **`--seed` also draws the validation window set** (see the `val_starts` comment in
  `train.py`), so absolute accuracies from different seeds are not comparable. Compare
  configurations *within* a seed and check the paired differences agree in sign. All v2
  sweep cells use seed 42, so they share windows and are internally comparable.
- **Val loss is much more sensitive than accuracy** and is the better tuning metric:
  lr 0.04 vs 0.08 at batch 32 differ by 0.0004 in accuracy but 0.0098 in val loss. But
  it also has a bigger replicate floor (0.0030), so neither metric rescues a 0.002 claim
  from a single run.

**Train loss is not a useful signal** for budget or capacity. With 48M table entries
against 2.16M training tokens the model can memorise: at 30 epochs train loss is 34%
lower than at 10 (0.0874 vs 0.1319) while val loss is 12% *worse* (0.1883 vs 0.1675) and
the generalisation gap triples (0.1009 vs 0.0356). Train loss flattening near the end of
a run is the cooldown forcing it, not saturation -- remove the cooldown constraint and it
keeps descending.

---

## Decision 1 -- budget in epochs, never in steps

`--max-steps` is not comparable across batch sizes. 200k steps is 3 passes over this
corpus at batch 1 and 96 at batch 32; and because the objective is a batch *mean*, batch
32 also took 32x the total parameter travel at the same LR. A step-budgeted sweep once
"found" a 0.016 batch-size effect that was entirely this (`HISTORY.md`). Under an epoch
budget, across batch 1 to 64 the spread within each architecture is 0.0009-0.0021, inside
the noise.

`--epochs` derives `max_steps = ceil(epochs * positions / (batch * CS))` in
`derive_schedule`, which also re-paces warmup, validation cadence and logging density
against the same axis.

---

## Decision 2 -- ten epochs, or eight

| budget | accuracy | wall clock | val loss at end vs its minimum |
|---|---|---|---|
| 3 | 0.9409 | 17 min | +0.0% |
| **10** | **0.9454** | **57 min** | +0.7% |
| 30 | 0.9459 | 2h46m | **+11.6%** |

Thirty epochs buys +0.0005 -- the noise floor -- for 3x the compute, and brings the
overfitting back: those runs peak at epoch 21-26 of 30 and then climb. Ten is where the
budget saturates.

`expE` then tested whether ten is *too* long. E=8 with `--lr-cooldown-frac 0.25` matches
E=10 to within the noise floor (-0.0004 accuracy, +0.0008 val loss, paired across two
seeds) at **21% less compute**. E=6 does not: -0.0020 accuracy and +0.0042 val loss,
consistent in sign across both seeds.

The runs plateau at about epoch 6 and the visible end-of-run improvement is the anneal,
not learning -- see Decision 5.

---

## Decision 3 -- peak LR scales linearly with batch size

Measured optima at **3 epochs**: 0.01 / 0.04 / 0.08 for batch 8 / 32 / 64. That is
exactly `0.00125 * batch`, three for three. Linear holds from batch 8 to 64 and breaks at
128 (which wants 0.08, not the 0.16 linear predicts) -- textbook critical-batch-size
behaviour.

**The base depends on the budget.** At 10 epochs batch 32 wants 0.08, i.e.
`0.0025 * batch`, because warmup is a fixed step count so a longer run anneals deeper on
its own. Using the 3-epoch rule at a 10-epoch budget costs ~6% val loss (0.1681 vs
0.1583) and doubles the generalisation gap (0.0356 vs 0.0186); higher LR means more
gradient noise, which suppresses memorisation. The v2 sweep was run at the 3-epoch rule
and is under-tuned for this reason.

LR response at batch 32, E=10:

| lr | 0.005 | 0.01 | 0.02 | 0.04 | **0.08** | 0.16 |
|---|---|---|---|---|---|---|
| accuracy | 0.9244 | 0.9390 | 0.9431 | 0.9450 | **0.9454** | 0.9394 |
| val loss | 0.3047 | 0.2236 | 0.1874 | 0.1681 | **0.1583** | 0.1700 |

**A wrong turn worth knowing:** linear scaling was once declared wrong for this model,
because extrapolating from batch 1's historical lr 0.01 gave 0.32 at batch 32, which
oscillated. The rule was right and the *anchor* was wrong: batch 1 wants ~0.00125 under
this schedule. Four LR grids bracketed the optimum from above only before this was
caught -- **when probing an LR, bracket from both sides.** Details in `HISTORY.md`.

---

## Decision 4 -- warmup = 6400 / batch

At a fixed epoch budget this holds `max_steps * batch` constant, so warmup covers the
same **204,800 tokens** *and* the schedule reaches the same depth
(`(W/max_steps)**0.5 = 0.178`) at every batch size.

A fixed step count does neither. With 200 steps everywhere, depth varies **elevenfold**
across batch 1 to 128 (0.031 / 0.089 / 0.178 / 0.252 / 0.356) -- and depth moves the LR
optimum, which is exactly why the 3-epoch and 10-epoch optima differ. Any batch-size
comparison at fixed warmup steps is partly a depth comparison.

Ramp length itself barely matters. With the decay start `t0` pinned by `--lr-decay-start`
(otherwise the warmup length also sets the height of the whole decay):

| W | 32 | 200 | 1000 |
|---|---|---|---|
| accuracy | 0.9383 | 0.9393 | 0.9354 |

32 and 200 are equivalent; 1000 is clearly worse.

---

## Decision 5 -- keep a terminal cooldown, do not tune its shape

**Having one is the single biggest win in this whole exercise.** `inverse-sqrt` never
anneals: it approaches `(warmup/max_steps)**alpha` and stops, leaving real LR on for
however long the run continues past its optimum. Without a cooldown, runs peaked as early
as **18% in** and then trained for hours, ending **15-27% above** their minimum val loss.
With one, that figure is 0.0-0.7% and runs peak at 88-100% of the budget.

**Beyond having one, nothing about it matters.** `expE`, 5 configs x 2 seeds, paired
within seed:

| comparison | accuracy | val loss | verdict |
|---|---|---|---|
| anneal length, 4 epochs vs 2 from an identical start | +0.0003 | +0.0003 | signs disagree -- no effect |
| anneal start, epoch 6 vs epoch 8 | -0.0001 | +0.0010 | signs disagree -- no effect |
| `--lr-cooldown-min-ratio` 0.05 vs 0 | -0.0002 | +0.0001 | signs disagree -- no effect |

This is a genuine null, not weak signal: the anneal was doing real work in those runs
(-0.0043 to -0.0082 val loss). The mechanism is visible in the data -- configs that start
annealing earlier, from a worse point, gain more, and **all finish in the same place**.
The anneal collects whatever gap remains and deposits the run on the same floor
regardless of when it begins or how long it takes.

---

## Decision 6 -- batch size is a speed knob only

At a token-matched schedule (warmup `6400/batch`, so depth and warmup tokens identical),
3 epochs:

| batch | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| best accuracy | 0.9281 | 0.9371 | 0.9409 | **0.9413** | 0.9341 |
| optimal lr | <0.005 | 0.01 | 0.04 | 0.08 | 0.08 |

Batch 32 and 64 are tied within noise and 128 falls off. In the v2 sweep, with LR scaled,
accuracy across batch 1 to 64 is flat within each architecture. Throughput saturates at
batch 8 and only batch 1 is penalised, at roughly half (`lut_torch/PERFORMANCE.md`,
"Training throughput by batch size").

---

## What the architecture sweep showed

`results/POS_torch_gpu_sweep_v2_E10/`, 32 cells, 27 complete.

- **`new-MHA=on` beats `new-MHA=off` by 0.010-0.017**, far outside every noise floor, and
  does it with **3-7x fewer parameters** (48-228M against 136M-1.68B). The smallest and
  fastest configuration is also the most accurate -- worth stating plainly because it is
  the opposite of the usual direction.
- **Nothing else in the sweep is a result.** The 15 `new-MHA=on` cells span 0.9435-0.9457
  -- 0.0022, smaller than the seed-to-seed floor. Head count, weight sharing and batch
  size are indistinguishable. Do not read that table as a ranking; `h8` holds both the
  best and the worst accuracy in the group, which is what scatter looks like.
- On matched hardware (`bs64`, both on a Quadro RTX 6000), **h8 costs 1.71x the runtime
  of h4** -- 89 min against 52 min -- for no measurable accuracy gain. Prefer h4.
- The `new-MHA=off` half is **under-tuned**: `expD` found its accuracy still rising at
  lr 0.08 where the rule gives 0.04, so at least +0.006 of the measured gap is tuning.
  The ranking is very unlikely to flip, but the margin is not yet a measurement.

## Still open (POS)

- **`new-MHA=off` LR is unbracketed.** `expD` ran 0.01/0.02/0.04/0.08 at batch 32 and was
  monotone increasing. Extend upward (0.08/0.16/0.32) before quoting the on-vs-off gap.
- **The 10-epoch LR rule (`0.0025 * batch`) is confirmed at batch 32 only.** Check batch
  8 (0.01 vs 0.02) and batch 64 (0.08 vs 0.16), ~4 GPU-h.
- **`E=8 + lr 0.08` has never been run together** with the chosen shape, though each half
  is validated. One confirming run before building on it.
- **Three cells cannot run**: `h8/mha=off/ws=full` is 1.68B parameters = 6.24 GiB of
  weights plus 6.24 GiB of dense gradient before activations, and the cluster mixes
  10.57 GiB to ~24 GiB cards behind one `--gres`. Pin the GPU by memory, or use
  `--table-grad sparse --cuda-graphs off`.
- **Everything is single-seed.** Differences under ~0.003 need 3+ seeds to mean anything.

---

# Part II -- the LM coded output head (full vocabulary)

This part is the **LM** task: wiki2 word-level, `V = 14,398`, same 6-layer body
(`n_t 64 / n_c 8 / ctx 32 / emb 64 / h4 / ws8 / new-MHA`, 47,972,352 body table entries),
batch 32, seed 42. The question is whether the output head can be replaced by `D`
coordinates plus a `(V, D)` code matrix -- `logits = y @ C.T + b` -- instead of one table
column per class. Six sweeps, ~280 GPU-h; the order they ran in is in `HISTORY.md`.

## The settled LM configuration

```
--output-code learned --output-code-bias
--output-code-dim 112               # settled at L6 (Decision L3); the depth runs used 448,
                                    # and the deep cell has not been repeated at 112
--num-layers 32                     # or 48 -- see Decision L5. n_c stays 8
--epochs 30                         # NOT more at depth: L24 at E=60 is worse than at E=30
--lr-scheduler cosine --cosine-min-lr-ratio 0.01
--lr-cooldown-frac 0.2              # every Part II run had this on top of cosine (HISTORY.md)
--warmup-frac 0.0095
--max-learning-rate 0.00125          # x batch 32 = 0.04 effective, coded head, E>=30
--lr-batch-scaling linear --lr-reference-batch-size 1
--validation-interval-epochs 0.1
```

Best measured: **`L60_nc8_lr04` (D448) at 107.46 perplexity** (min 106.88; acc 0.2631,
no-unk ppl 133.1), 493.5M parameters, 26.2 h. The recommended point is **`L32` at 113.53**
(269.7M, 14.1 h) -- the deepest cell still smaller than the dense baseline -- or `L48` at
109.20 (397.6M, 21.6 h) if parameters are not the constraint.

Best dense baseline: **`dense_E60_lr04` at 120.65** (283.9M, 8.8 h).

**The headline is "deeper coded models beat the dense head at equal parameters", not "the
coded head is smaller at equal wall clock".** At 6 layers the dense head beats every coded
run; the coded head wins only when its freed parameters go into depth. All single-seed:

```
equal parameters   L32 coded  269.7M  14.1 h  113.53   vs  dense E60  283.9M  8.8 h  120.65   (-5.9%)
equal wall clock   L18 coded  157.7M   8.0 h  120.42   vs  dense E60  283.9M  8.8 h  120.65   (tie, -44% params)
```

The coded model costs 1.6x the wall clock at equal parameters (depth is serial), and wins
nothing at equal wall clock except size. See Decisions L4 and L5.

## Decision L1 -- the coded head is a real replacement, and `learned` is the only version that works

At `E=10`, peak 0.04, the arms that answer the question:

| head | D | ppl | acc | head params |
|---|---|---|---|---|
| dense (`V` logits) | -- | 184.7 | 0.2314 | 235,896,832 |
| learned | 28 | 187.7 | 0.2375 | 876,294 |
| learned | 14 | 218.6 | 0.2333 | 445,346 |
| **random** (frozen Gaussian) | 28 | 399.7 | 0.2259 | 473,150 |
| **binary** (frozen ECOC) | 14 | 403.3 | 0.1802 | 243,774 |

A learned `D = 28` matches the dense head to within 1.6% perplexity at **269x fewer head
parameters**. `D = ceil(log2 V) = 14` is not enough: 18% worse than 28.

**Freezing the code is what costs, not the code's Hamming distance.** Frozen random at
`D = 28` (399.7) is indistinguishable from frozen ECOC at `D = 14` (403.3), although the
random code's minimum distance is far larger. Widening `D` does not rescue a frozen code.

## Decision L2 -- 30 epochs with a cosine anneal; `inverse-sqrt` is the wrong shape here

Unlike POS, the LM task is **not** saturated at 10 epochs -- it is budget-starved. All at
`D = 28` learned + bias:

| arm | schedule | epochs run | ppl | ppl @ ep 23.5 |
|---|---|---|---|---|
| (E10 reference) | inverse-sqrt, cooldown 0.2 | 10 | 187.7 | -- |
| `A1_E30_depthfree` | inverse-sqrt, warmup left at 200 steps | 23.5\* | (167.3) | 167.3 |
| `A2_E30_depthpin` | inverse-sqrt, `--warmup-frac 0.0095` | 30 | 152.6 | 157.4 |
| `C3_E30_cooldown01` | inverse-sqrt, cooldown 0.1 | 30 | 151.9 | 157.1 |
| `C1_E30_alpha025` | inverse-sqrt `alpha 0.25`, depth-pinned | 23.5\* | (150.1) | 150.1 |
| **`C2_E30_cosine`** | **cosine to 1%, depth-pinned** | **30** | **144.0** | **145.7** |
| `A3_E60_depthpin` | inverse-sqrt, depth-pinned | 60 | 138.0 | 150.8 |

\* stopped early and never annealed; compare all arms in the last column.

- **Cosine beats every `inverse-sqrt` variant at the same budget**, by 5.6% at 30 epochs
  (144.0 vs 152.6) and by 3.0% at the common epoch 23.5 (145.7 vs 150.1 for the best
  `inverse-sqrt` arm). This is the opposite of the POS finding in Decision 5, where the
  cooldown's shape was a measured null. The regimes differ: POS overfits and the anneal only
  collects a gap that is already there, while the LM task is still descending at the end of
  the budget, so the whole decay curve does real learning and its shape matters.
  `alpha 0.25` (`C1`) is the runner-up and beats `alpha 0.5` by 4.7% at epoch 23.5 --
  consistent with cosine's advantage being "hold a higher LR for longer, then anneal hard".
- **`A3`'s 60-epoch win is budget, not shape.** At epoch 30 `A3` is at 146.6, *behind*
  `C2`'s 144.0 at the same point; it only passes it by spending twice the compute.
- **Warmup must be pinned as a fraction, not a step count.** `--warmup-frac 0.0095`
  reproduces the 200-step depth of the 10-epoch schedule at 30 epochs; leaving warmup at 200
  absolute steps costs 6.3% at matched epoch (167.3 vs 157.4). Same mechanism as POS
  Decision 4 -- in `(W/s)**alpha` the warmup length sets the height of the whole decay -- but
  here it is worth 10 perplexity rather than nothing.

`--max-learning-rate 0.005` (0.16 effective) diverges to NaN. The probe at `E=10` gave
0.01 -> 255.2, 0.02 -> 213.7, 0.04 -> 187.7, 0.08 -> 171.9, 0.16 -> NaN.

## Decision L3 -- `D = 112`, and never `binary`

`LM_torch_code_width`, all 30 epochs at the C2 schedule, seed 42. `D = 28` is the C2 run
(command-line identical apart from paths). "gap" is val loss minus train EMA loss.

| D | learned ppl | binary ppl | learned gap | binary gap | head params | vs dense |
|---|---|---|---|---|---|---|
| 14 | 157.3 | 394.9 | -0.02 | -0.05 | 445,346 | 530x |
| 28 | 144.0 | 320.0 | -- | -0.05 | 876,294 | 269x |
| 56 | 135.0 | 278.1 | +0.11 | -0.04 | 1,738,190 | 136x |
| **112** | **133.4** | 253.0 | +0.22 | -0.01 | 3,461,982 | **68x** |
| 224 | 133.1 | 229.7 | +0.32 | +0.01 | 6,909,566 | 34x |
| 448 | 132.0 | 204.5 | +0.39 | +0.02 | 13,804,734 | 17x |
| 1000 | 131.9 | 181.8 | +0.47 | +0.06 | 30,796,398 | 7.7x |

**Learned saturates at `D = 112`.** From 112 to 1000 is 9x the head parameters for 1.1%
perplexity. The gap column shows why: learned's train/val gap grows monotonically from
-0.02 to +0.47, so everything past `D = 112` goes into memorising. `D = 56` is the cheap
option at 1.2% worse; below that the loss is real (`D = 28` is +8%, `D = 14` is +18%).
`D = 112` over `D = 224` is the smaller of two tied arms, not a measured winner.

**`binary` never saturates because it never fits.** Its gap stays at roughly zero across
the entire range -- it is still underfitting at `D = 1000`. It is not a regulariser; the
frozen code is a capacity and optimisation bottleneck. **Learned `D = 14` (157.3) beats
binary `D = 1000` (181.8)**, a >70x width disadvantage with no crossover anywhere.

`binary` also saves no storage at equal `D` (the frozen `C` is a buffer, absent from the
parameter count but still `V*D` floats), and its occasional raw-accuracy edge is the bias
capturing `<unk>` -- on `acc_no_unk` it loses.

Do not spend further GPU time on `binary`, and **do not fix `ecoc_masks` first and
re-test**. Its minimum Hamming distance is stuck at 2 for every `D` from 28 to 1000 (a
greedy-construction defect), but Decision L1 shows distance is not the binding constraint.

## Decision L4 -- peak LR 0.04 at E>=30; at L6 the dense head wins

`LM_torch_lr_budget`, all cosine to 1%, `--warmup-frac 0.0095`, seed 42. `D = 112` learned
+ bias. The `E=30 / 0.04` cell is `L_D112` from the width sweep (identical command line
apart from `--epochs`). Final-step perplexity:

| `D = 112` | peak 0.04 | peak 0.08 |
|---|---|---|
| E = 30 | 133.38 | 142.38 |
| E = 60 | 126.52 | 137.20 |
| E = 120 | **125.38** (min 124.57 @ ep 114) | 132.59 |

| dense (`V` logits) | peak 0.04 | peak 0.08 |
|---|---|---|
| E = 30 | 132.29 | **128.39** |
| E = 60 | **120.65** (min 119.93 @ ep 59) | *no result* |

**At `L6` the coded head is smaller but worse at any matched budget.** The dense head at
E=60 (120.65 in 8.8 h) beats every `L6` coded run, including `D112_E120_lr04` (125.38 in
10.8 h). The coded head's advantage comes from spending the freed parameters on depth
(Decision L5), not from the head alone.

**The coded head's budget is exhausted near E=60.** E=60 -> E=120 at 0.04 is 126.52 ->
125.38, 0.9% for 2x compute, with the train/val gap at +0.64. More epochs at `L6` /
`D = 112` are not worth running.

**Re-bracket LR whenever the budget changes.** The `E=10` probe preferred 0.08 by 8.4%; at
30 epochs 0.08 *loses* by 6.7% and at 60 by 8.4% (same budget-depth mechanism as POS
Decision 3 -- a longer run anneals deeper on its own, so it wants a lower peak). At E=30
the two heads want opposite LRs: coded prefers 0.04, dense prefers 0.08 (128.39 against
132.29), so a dense control at one LR only would have been wrong whichever LR was chosen.
At E=60 the dense preference appears to move to 0.04 too (123.90 against 125.23 at epoch
37.5, the last common point). Nothing here diverged, including 120 epochs at 0.08.

**At 0.08 the coded model converges to ~132 regardless of budget** (137.20 at E=60, 132.59
at E=120, flat over its last 10 epochs: 132.55 -> 132.59). At 0.04 it goes lower but
saturates by E=120 too.

**Overfitting is the constraint.** Train/val gap at `D = 112` / 0.04 (val loss minus train
EMA loss): +0.217 at E=30, +0.383 at E=60, +0.637 at E=120 (+0.583 at E=120 / 0.08). The
dense head's gaps are far smaller: +0.07 and +0.15 at E=30, +0.22 at E=60 / 0.04. The coded
head memorises more readily, so more budget wants more dropout or more data with it, not
epochs alone.

## Decision L5 -- depth is where the freed budget goes; `n_c` is not

`LM_torch_depth`, learned `D = 448` + bias, cosine to 1%, `--warmup-frac 0.0095`,
30 epochs, seed 42. Full factorial over layers x `n_c` x peak LR, 18 cells. `n_c` raises
both `--n-c` (FFN / MHA_O / V / Final) and `--attn_qk_n_c` (Q, K), so 8 -> 10 is 4x the
rows in every table.

Validation perplexity:

| | 0.04 / n_c=8 | 0.04 / n_c=10 | 0.08 / n_c=8 | 0.08 / n_c=10 |
|---|---|---|---|---|
| L6 | 130.1 | 135.2 | 142.9 | 152.9 |
| L12 | 123.2 | 130.7 | 136.0 | 141.8 |
| L18 | 119.2 | 123.7 | 130.4 | 134.6 |
| L24 | **113.9** | 119.0 | 127.6 | 129.5 |
| dense L6 | 131.7 | -- | 127.9 | -- |

**Depth is monotone in all four ladders.** The deepest cell wins every column. Four
independent ladders agreeing matters more than the size of any one gap.

**The deeper ladder** (`20260918-103308_*`, `run_files/torch_LM_deeper_gpu.sh`) extended the
winning column (`n_c=8 / 0.04`, same flags apart from `--num-layers`). Min val perplexity
(finals in brackets), with the train/val gap computed from the final val loss:

| layers | ppl | total params | wall clock | gap | step gain |
|---|---|---|---|---|---|
| L6 | 130.1 (130.69) | 61.8M | 2.4 h | -- | -- |
| L12 | 123.2 (123.80) | 109.7M | 4.8 h | -- | -5.3% |
| L18 | 119.2 (120.42) | 157.7M | 8.0 h | +0.410 | -3.2% |
| L24 | 113.9 (114.45) | 205.7M | 10.7 h | +0.405 | -4.5% |
| L32 | 113.2 (113.53) | 269.7M | 14.1 h | +0.422 | -0.6% |
| L48 | 108.8 (109.20) | 397.6M | 21.6 h | +0.453 | -3.9% |
| **L60** | **106.9 (107.46)** | 493.5M | 26.2 h | +0.463 | -1.7% |

**Depth still pays at 60 but has started to flatten**, from ~5% per 6 layers early to
~1.7% for the last 12, while cost is linear in layers. The `L24 -> L32` step (-0.6%) is out
of line with its neighbours. It is probably seed noise, which is unmeasured, not a
plateau, since L48 resumes the trend. The train/val gap rises only slowly with depth
(+0.41 -> +0.46), so the extra layers are mostly going to generalisation, not
memorisation. Lower layers show no starvation through L60.

**`L32` is the last depth smaller than the dense baseline** (269.7M against 283.9M). Past
it the claim changes from "better and smaller" to "better". `L48` and `L60` are 1.4x and
1.7x the dense model.

**Budget does not pay at depth: `L24_E60` (119.82, min 119.50) is worse than `L24` at E=30
(114.45).** With the same cosine shape stretched over 60 epochs, the run is at 122.7 at
epoch 30 and only reaches 119.5 near the end. Its train/val gap is +0.768 against +0.405,
so the extra 30 epochs went into memorisation. This matches Decision L4 at `L6`: on this
data, budget past ~30 epochs feeds overfitting, and depth is the axis that still
generalises. Do not raise epochs at depth without regularisation. If anything, try
*fewer* (E=15-20) at L32/L48, which would also cut their wall clock.

**`n_c = 10` is worse than `n_c = 8` at every matched depth**, by 4-10%, at both LRs. Four
times the rows per table, reliably worse -- so the exponential axis is not the one to
spend on, and `n_c = 8` stands.

**The matched-budget pair settles depth against capacity.** `L24 / n_c=8` and
`L6 / n_c=10` are both exactly 205.7M entries:

```
L24 n_c=8   ->  113.9      depth
L6  n_c=10  ->  135.2      capacity
```

16% better perplexity for identical parameters. A per-layer table is 7,995,392 entries, so
the ~222M the coded head frees buys ~27 layers, and they are worth more than the same
bytes as table rows.

**Against the best dense run** (`dense_E60_lr04`, 120.65, Decision L4):

```
L24 coded   205.7M  10.7 h  114.45   -5.1%  at 28% fewer params
L32 coded   269.7M  14.1 h  113.53   -5.9%  at  5% fewer params
L18 coded   157.7M   8.0 h  120.42   tie    at 44% fewer params, equal wall clock
```

Depth at E=30 is still worth more than 2x epochs at `L6`: `L24` beats `D112 / E=60 / 0.04`
(125.5) and `D112 / E=120 / 0.04` (125.38). The dense head has not been given depth, so this
compares "coded + freed layers" against "dense at 6 layers", which is the intended trade
but not a same-depth comparison.

**Decision L4's E=30 LR split replicates independently.** Coded cells prefer 0.04 in all
eight pairs; the dense baseline prefers 0.08 at E=30 (127.9 against 131.7).

**First LM replicate floor: ~0.11%.** `base_L6_lr08` here and `dense_E30_lr08` in
`LM_torch_lr_budget` are byte-identical command lines at seed 42 -- verified flag by flag --
and landed on 127.87 and 128.01, 0.14 ppl apart. CUDA's `index_add_` atomics are not
deterministic, so that is the run-to-run floor for a *repeat*, not for a seed change; seed
variance is still unmeasured and will be larger. It does mean the `n_c` and depth effects
here (4-16%) are two orders of magnitude clear of noise, while the `L24 n_c=8` vs
`L24 n_c=10` accuracy tie (0.2548 vs 0.2554) is not.

**Caveats.** Run at `D = 448`, not the `D = 112` of Decision L3 -- chosen before L3 settled,
and it should be repeated at 112 before `D` and depth are quoted together. Single seed
throughout.

## Decision L6 -- the nanoGPT baseline: best no-unk ppl 96.9, and a 1.2M-parameter GPT ties L60

`GPT_LM_baseline` (`gpt_baseline_train.py`, `run_files/GPT_LM_baseline_gpu.sh`): an
unmodified nanoGPT `GPT` with data, validation windows, `evaluate()` and the epoch/cosine
schedule taken from `lut_torch`. Same wiki2 10-freq data (`V = 14,398`), ctx 32, batch 32,
30 epochs, cosine to 1%, `--warmup-frac 0.0095`. The optimizer is the transformer's own
(AdamW 0.9/0.95, wd 0.1, clip 1, bf16). Seed 42 and `--testing-length 1000`, so these runs
score **the same 1,000 windows as the `LM_torch_depth` runs** (`test_gpt_baseline.py`
asserts the window equality), and the `loss.csv` files compare directly. Single seed.

Final val (min in brackets where it differs), all 12 cells:

| cell | params | ppl | no-unk ppl | wall clock |
|---|---|---|---|---|
| **d256 / L12 / do0.3** | 13.1M | **81.38** | **96.93** (96.74) | 0.34 h |
| d512 / L6 / do0.3 | 26.3M | 82.96 (82.93) | 97.63 (97.24) | 0.32 h |
| d256 / L6 / do0.1 | 8.4M | 83.11 | 97.89 (97.77) | 0.16 h |
| d256 / L12 / do0.1 | 13.1M | 85.69 | 101.70 (101.02) | 0.23 h |
| d512 / L12 / do0.3 | 45.1M | 83.80 | 98.50 (98.26) | 0.45 h |
| d256 / L6 / do0.3 | 8.4M | 88.29 | 104.58 (104.46) | 0.16 h |
| d128 / L6 / do0.1 | 3.0M | 89.69 | 107.91 | 0.49 h |
| d512 / L6 / do0.1 | 26.3M | 96.30 (93.26) | 109.68 (107.41) | 0.26 h |
| d128 / L6 / do0.3 | 3.0M | 104.12 | 125.87 | 0.48 h |
| d64 / L6 / do0.1 (LUT shape) | 1.2M | 108.19 | 132.37 | 0.41 h |
| d64 / L6 / do0.3 | 1.2M | 129.83 | 160.36 | 0.50 h |
| d512 / L12 / do0.1 | 45.1M | 130.11 (97.01) | 142.97 (115.14) | 0.37 h |
| *LUT `L60_nc8_lr04`* | *493.5M* | *107.46* | *133.07* | *26.2 h* |
| *LUT `L32_nc8_lr04`* | *269.7M* | *113.53* | *140.99* | *14.1 h* |

**Size saturates at ~8M parameters; past that, more transformer does not help.** Best
cell per size: 1.2M -> 108.2, 3.0M -> 89.7, 8.4M -> 83.1, 13.1M -> 81.4, 26.3M -> 82.9,
45.1M -> 83.8. From 8M to 45M (5.4x) the spread is 2.7 ppl, about the size of the
dropout effect inside one size. The dropout pattern is the classic data-limited signature:
small models want 0.1 (do0.3 costs d64 +21.6 and d128 +14.4), large ones need 0.3 (do0.1
costs d512/L6 +13.3, and d512/L12 overfits from 97.0 at epoch 11.8 to 130.1). The optimal
regulariser grows with size while the best score stays flat. A larger transformer is not
the missing piece; more data (or longer context) is.

**Why 96.9 reads high.** Every position of a 32-word window is scored, and early positions
have little context (ppl 202 at position 0, ~67-80 from position 8; positions >= 16 alone
give 72.5 / no-unk 86.2). No-unk also removes the easy `<unk>` targets (8.8% of val, ppl
13.2), and published WT2 figures use a 33k vocabulary and carried state, so they are not
comparable; treat the with-unk 81.4 as a sanity range, not a benchmark.

**The data limits quantity, not quality.** 2.16M training tokens is what limits the larger
models: every large or lightly-regularised cell overfits (d512/L12/do0.1 bottoms at 97.0 at
epoch 11.8 and ends at 130.1; d512/L6/do0.1 peaks at epoch 20.6), while the best cells are
the high-dropout ones and are still improving at epoch 30. That is a data-size ceiling,
consistent with `L24_E60` memorising in Decision L5. Nothing points to a *quality* problem:
the tokens are verified identical to the LUT stream, and the per-position curve has the
shape expected of clean text. WikiText-103 (Part III) is the test that separates "small
data" from "our models", for both model families.

**What matters for the LUT story is the gap, not the absolute number.** On identical windows:

```
GPT d64/L6 (LUT body shape)    1.2M params  0.41 h   108.19 / no-unk 132.37
LUT L60_nc8_lr04             493.5M params  26.2 h   107.46 / no-unk 133.07
GPT best (d256/L12/do0.3)     13.1M params  0.34 h    81.38 / no-unk  96.93
```

**A transformer with the LUT body's shape and 1/400 of its parameters ties the best LUT.**
The tuned transformer is 24% better in ppl (27% in no-unk), with 1/38 of the parameters,
in 1/77 of the wall clock. The LUT's "better at equal parameters" result (Decision L5) is
measured against a LUT dense head, not against a transformer. Quote it that way.

## Still open (LM)

- **Seeds.** Everything is seed 42. The *repeat* floor is ~0.11% (Decision L5), but seed
  variance is unmeasured and the POS floors do not transfer. The `L24 -> L32` step, the
  L18-vs-dense-E60 tie and `D = 112` vs `D = 224` all need seeds before they mean anything.
  Cheapest: two extra seeds for L24, L32, L48 and dense E60; for the GPT baseline, seeds
  43/44 for the best cell and d64/L6 (array indices +12, +24).
- **Repeat the depth result at `D = 112`.** The depth grid ran at `D = 448`; repeat `L32`
  (and ideally `L48`) at 112 before quoting `D` and depth together. A narrower head may
  also overfit less, which is now the binding constraint.
- **Budget and regularisation at depth, not more depth.** L60 is best but the last 12
  layers bought 1.7% for +24% wall clock. Try E=15-20 and/or higher dropout at L32/L48.
- **GPT baseline tuning.** The best cells were still improving at epoch 30, so the
  baseline is probably under-, not over-, estimated. Try dropout 0.4-0.5 at d256-d512.
- **`dense_E60_lr08` has no result.** Rerun it, or accept 0.04 on the epoch-37.5 lead.

---

# Part III -- WikiText-103 with GPT-2 BPE (decided 2026-10-14, runs pending)

Decision L6 left two questions: is the WT2 LM data-limited, and is the tokenizer part of
the problem? This part records the decisions made before running anything, so the
reasoning survives whatever the numbers turn out to be.

## Decision W1 -- the tokenizer was not the problem, but the LM line leaves it anyway

The WT2 source is already tokenized (v1), and `word_tokenizer.py` re-tokenizes it with
`min_freq 10`, more than doubling `<unk>` and splitting `@-@` / `'s`. Those artifacts are
easy targets (13.4% of targets; on the remaining real words the best GPT is at ppl 117.9
against 81.4 overall), so the tokenizer flatters the headline, but it cannot explain the
GPT/LUT gap because both see the identical stream.

Decided:

- **LM (both GPT and LUT) uses GPT-2 BPE (tiktoken `gpt2`, V = 50,257) on the *raw*
  WikiText-103 text.** It is open-vocabulary (no `<unk>`), standard, needs no new
  dependency, and is the only option whose vocabulary fits at WT103 scale. WikiText-native
  word tokens were rejected: V = 267,735 at WT103 needs an adaptive softmax.
- **POS keeps the word tokenizer unchanged.** The spaCy alignment in
  `WordTokenizer._encode_pos` is built on it.
- **Known interaction, set aside for now:** the LUT's input embedding is frozen random
  vectors (`LutModel.token_embedding`), while the GPT learns its own. Under BPE the LUT
  has to compose words from random subword vectors. A learned input embedding for the LUT
  is a separate experiment, deliberately postponed.

## Decision W2 -- the headline metric is EXACT per-word perplexity

BPE perplexity is per *token* (~1.19 GPT-2 tokens per word on this text), so it is lower
than a per-word perplexity of the same model, and it cannot be set next to word-level
numbers. The total NLL of a text does not depend on how the text is cut into tokens, so
dividing it by the number of words gives a tokenizer-independent per-word perplexity.

**We use the exact version, computed inside `evaluate()` for every validation**, not a
global tokens/words ratio applied afterwards:

```
loss_per_word = (sum of token NLL over all scored targets)
              / (number of scored targets that start a word)
```

A target starts a word iff its bytes contain a non-whitespace byte AND (it begins with
whitespace OR the previous token ends with whitespace OR it is the first token of the
stream). Over a whole stream this count equals `len(text.split())` exactly, and a test
asserts that. Whitespace-only tokens (`"\n"`) count in the numerator, not the
denominator. A word that starts before a window contributes its tail tokens to the
numerator but no word to the denominator, which is exact for the targets actually scored.
The global-ratio alternative was rejected because 1,000 sampled windows do not have the
corpus-average tokens/word.

Consequences:

- `loss.csv` gains two columns at the end, `loss_per_word, perplexity_per_word`, in both
  trainers. They are NaN for the word tokenizer. Read BPE runs by `perplexity_per_word`.
- The `_no_unk` triple is NaN under BPE. It used to test `target == 0`, and in GPT-2 BPE
  id 0 is `"!"`, so leaving it on would silently drop every `!`.
- Per-word numbers are exact *within* the BPE pipeline (GPT vs LUT, the data-size
  ladder). Against the old WT2 word-tokenizer numbers they compare only roughly, because
  the vocabulary, `<unk>` and text variant all differ.

## Decision W3 -- pre-tokenized `.bin` input, frozen once (loader option B)

`prepare_wikitext103.py` downloads `wikitext-103-raw-v1.zip` (the zip mirror, chosen over
Hugging Face parquet to avoid a new `pyarrow` dependency). It writes raw splits, nested
training subsets (2M / 8M / 32M words, cut at article boundaries) and uint16 GPT-2 `.bin`
token files to `nanoGPT_POS/data/wiki103/`. Each `.bin` has a `<file>.json` manifest
(token and word counts, source and token SHA-256, tiktoken version).

Both trainers accept a `.bin` path wherever they accept a text file. `load_datasets`
memory-maps it after validating the manifest, and the text path is unchanged. Rejected:

- **A, encode at startup:** ~1.5 min and ~6 GB per job, and the token stream would depend
  on whichever tiktoken a node has.
- **C, automatic cache:** 12 array jobs starting together race to write it.

The run scripts copy both manifests into `config.txt`, so every run records the exact token
stream it trained on.

WikiText-2 and -103 share the validation articles. The prep script checks this against
the WT2 val file and records the result in `val_gpt2.bin.json`. Same val `.bin` + same
seed means the same windows for every run in this part.

## Decision W4 -- two waves, both trainers built together

One implementation pass for both trainers, then:

- **Wave 1:** the GPT data-size x model-size ladder (`run_files/GPT_LM_wt103_gpu.sh`, 12
  cells: {2M, 8M, 32M, full} x {d256/L12, d512/L12, d768/L12}, epochs 30/15/5/2 and
  dropout 0.3/0.2/0.1/0.1 by subset). Plus a LUT L6 smoke at 0.1 epoch
  (`run_files/torch_LM_wt103_gpu.sh` arm 0), which catches LUT pipeline problems early.
- **Wave 2,** after wave 1 is checked: LUT `L6_2M_E30` (the WT2-sized bridge) and
  `L6/L18/L32` for 1 epoch of full WT103 (arms 1-4). L32 is ~1-1.5 days.

The question wave 1 answers: if bigger transformers start winning as data grows, WT2 was
data-limited (Decision L6). Wave 2 asks the same of LUT depth, which at WT2 scale
memorised (`L24_E60`).

## Result W5 -- wave-1 GPT: the WT103 numbers are high because of rare words, not BPE

Status on 2026-10-14 16:45, seed 42. The headline is per-word ppl (final, with the min in
brackets where it differs). Runtime is training time from the tqdm bar. **It is only
comparable on the same GPU type**: the cluster mixes RTX 2080 Ti, A10 and A5000, and
`full_d512` on an A5000 (1.37 h) ran faster than `full_d256` on a 2080 Ti (3.20 h).

GPT (`GPT_LM_wt103`): AdamW, ctx 32, batch 32, cosine to 1%, warmup 0.95%, L12.

| run | data | epochs | dropout | peak LR | params | GPU | runtime | ppl/word | status |
|---|---|---|---|---|---|---|---|---|---|
| 2M d256 h8 | 2M | 30 | 0.3 | 1e-3 | 22.3M | 2080 Ti | 1.05 h | 496.9 (496.5) | done |
| 2M d512 h8 | 2M | 30 | 0.3 | 6e-4 | 63.5M | 2080 Ti | 1.67 h | 494.1 (493.7) | done |
| 2M d768 h12 | 2M | 30 | 0.3 | 6e-4 | 123.6M | 2080 Ti | 0.85 h | 774.0 at ep 8.1 | crashed (ckpt dir) |
| 8M d256 h8 | 8M | 15 | 0.2 | 1e-3 | 22.3M | 2080 Ti | 1.88 h | 370.2 (369.6) | done |
| 8M d512 h8 | 8M | 15 | 0.2 | 6e-4 | 63.5M | 2080 Ti | 3.26 h | 307.0 (306.5) | done |
| 8M d768 h12 | 8M | 15 | 0.2 | 6e-4 | 123.6M | 2080 Ti | 0.43 h | 838.8 at ep 1.05 | crashed (ckpt dir) |
| 32M d256 h8 | 32M | 5 | 0.1 | 1e-3 | 22.3M | 2080 Ti | 2.43 h | 302.2 (302.1) | done |
| 32M d512 h8 | 32M | 5 | 0.1 | 6e-4 | 63.5M | 2080 Ti | 4.16 h | 238.2 (237.9) | done |
| 32M d768 h12 | 32M | 5 | 0.1 | 6e-4 | 123.6M | 2080 Ti | 4.9 h, 64% | 456.7 at ep 3.2 | running |
| full d256 h8 | 103M | 2 | 0.1 | 1e-3 | 22.3M | 2080 Ti | 3.20 h | 288.1 | done |
| full d512 h8 | 103M | 2 | 0.1 | 6e-4 | 63.5M | A5000 | 1.37 h | 225.6 | done |
| full d768 h12 | 103M | 2 | 0.1 | 6e-4 | 123.6M | -- | -- | -- | not started |

LUT (`LM_torch_wt103`): the settled body (emb 64, h4, n_t 64, n_c 8, ws 8, new-MHA),
learned code D112 + bias, SGD peak 0.04, cosine with cooldown 0.2, dropout 0.2. Params =
table entries + output code.

| run | data | epochs | params | GPU | runtime | ppl/word | status |
|---|---|---|---|---|---|---|---|
| L6 smoke | 103M | 0.1 | 55.5M | A5000 | 0.46 h | 1059.9 at ep 0.09 | done |
| L6 2M E30 | 2M | 30 | 55.5M | A5000 | -- | -- | crashed at first save (ckpt dir) |
| L6 | 103M | 1 | 55.5M | A5000 | 4.45 h | 545.2 (543.4) | done |
| L18 | 103M | 1 | 151.4M | A10 | 4.9 h, 33% (~15 h total) | 655.4 at ep 0.33 | running |
| L32 | 103M | 1 | 263.4M | A5000 | 4.8 h, 22% (~22 h total) | 689.9 at ep 0.21 | running |

The three "crashed (ckpt dir)" runs died with "Parent directory ... does not exist" on
`/cs/snapless`, although the script had created it. Both trainers now re-create the
directory before every save.

**d768 at peak LR 6e-4 is unstable.** At equal epochs it is worse than d256 in all three
runs (2M at ep 8.1: 774 against 704; 8M at ep 1.05: 839 against 662; 32M at ep 3.2: 457
against 394). On 32M its loss *rose* from 830 to 1,025 between epochs 0.3 and 0.6, while
no d256 or d512 run rose more than 3.5% above its running best. nanoGPT's 6e-4 for this
width assumes ~0.5M-token batches; ours are 1,024 tokens. Rerun d768 at a lower peak LR,
not as is.

- **Data-limited, confirmed.** At 2M words the two widths tie; from 8M up, d512 wins by
  17-22%. Every cell from 8M up has its minimum at the last validation, and the full-data
  cells gained 4% in their last 10%, so 2 epochs on full data is too short.
- **Why 494 at 2M when WT2 gave 81 on the same text.** The val sets are word-for-word
  identical, so each whitespace word can be classified by whether the WT2 pipeline
  (`vocab_file_10_freq`) could predict it. 90.7% of words can. The BPE model's per-word
  loss is the sum of its tokens' NLL. Scored on the 1,000 run windows, words that fall
  wholly inside a window:

  | checkpoint | all words | WT2-vocab words (90.8%) | other words (9.2%) |
  |---|---|---|---|
  | 2M_d512 | 493.7 | **128.7** | 1.4e8 |
  | 32M_d512 | 237.9 | **75.3** | 1.0e7 |
  | full_d512 | 225.6 | **72.7** | 8.4e6 |

  On the words WT2 could predict, the 2M BPE model is at 128.7, against WT2's 117.9 for
  real words + punctuation (Decision W1). That is the same ballpark: a 32-token BPE window
  holds ~27 words, and the model also has to learn spelling. The whole gap to 494 is the
  9.2% of words WT2 mapped to `<unk>`, which BPE has to spell out (~16-19 nats each). So
  the high headline is the honest cost of an open vocabulary, not a pipeline fault. With
  full data, the same words reach **72.7**, well below anything WT2-sized achieved.
  The WT2 `no_unk` columns are *not* this comparison: they drop the `<unk>` targets but
  still compare against a closed 14k vocabulary.
- Script: the same-words analysis is a one-off (no tool in the repo). Its rule: whitespace
  word w is "WT2-vocab" iff every `word_tokenizer` regex piece of w is in
  `vocab_file_10_freq.txt`; words cut by a window edge are dropped; whitespace-only tokens
  belong to no word.

Follow-up: `run_files/GPT_LM_wt103_depth_gpu.sh`. Full data, 4 epochs, d256/d512 x L12/L24.

---

## Run index

| directory | what it is |
|---|---|
| `results/POS_torch_gpu_sweep/` | first sweep, 24 cells, fixed 200k steps, lr 0.01, no cooldown |
| `results/POS_torch_lr_probe/`, `lr_probe2/` | first LR probes; stage 1 used base-lr x scaling, stage 2 switched to absolute LR |
| `results/POS_torch_expA/`, `expA2/` | peak LR x budget at batch 32, and closing that grid |
| `results/POS_torch_expB/` | schedule shape: ramp, decay start, cooldown fraction |
| `results/POS_torch_expC/` | batch size at a token-matched schedule |
| `results/POS_torch_expD/` | does the LR rule transfer to `new-MHA=off` |
| `results/POS_torch_expE/` | cooldown shape vs budget, 2 seeds, paired |
| `results/POS_torch_gpu_sweep_v2_E10/` | the rerun under this recipe, 32 cells |
| `results/LM_torch_output_code_E10/` | LM, does the coded head work: dense baseline, learned D14/D28/D64, binary D14 |
| `results/LM_torch_schedule_probe/` | LM, budget and LR shape at D28: `A*` budget/warmup, `B*` peak LR, `C*` anneal shape, `R*` frozen-random controls |
| `results/LM_torch_code_width/` | LM, `D` in {14,56,112,224,448,1000} x {learned, binary} at the C2 schedule, 30 epochs. Learned `D28` is `C2_E30_cosine` in the schedule probe |
| `results/LM_torch_depth/` | LM, layer count x `n_c` x peak LR at learned D448, 30 epochs, 18 cells, plus the deeper ladder (`20260918-103308_*`: L18 refill, L32, L48, L60, `L24_E60`). Decision L5 |
| `results/LM_torch_lr_budget/` | LM, peak LR x budget at D112 plus dense controls at E=30 and E=60 (round 2: `20260918-*`). Decision L4 |
| `results/GPT_LM_baseline/` | LM, nanoGPT baseline (`gpt_baseline_train.py`): width {64..512} x depth {6,12} x dropout {0.1,0.3}, seed 42, same windows as the LUT LM runs. Decision L6. Arm label is in the directory name and `config.txt` (`arm=`) |

Crashed, preempted and never-run cells are listed in `HISTORY.md` ("Infrastructure
incidents"). Each run's `config.txt` records its swept axis under `swept_*` keys and the
full command; trust those over directory names.
