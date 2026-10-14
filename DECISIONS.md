# DECISIONS.md

Why the training configuration is what it is. `AGENTS.md` and `lut_torch/AGENTS.md` carry
the verdict; this file carries the measurements behind it, so that a later change can see
which numbers it would be overturning rather than rediscovering them.

Part I is the POS task; **Part II covers the LM coded output head** and its conclusions
do not transfer in either direction -- where they conflict (anneal shape, budget), the
regimes differ and both are correct for their own task.

Part I measured on the POS task (wiki2, 2,156,877 training positions, context 32) at
`heads 4 / new-MHA on / weight-sharing 8`, batch 32, unless stated otherwise. About
40 GPU-h of probes (`expA`-`expE`, two LR probe stages) plus two full sweeps.

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
32 also took 32x the total parameter travel at the same LR.

The first GPU sweep (`results/POS_torch_gpu_sweep/`) was budgeted that way, and its
headline result -- batch 32 beating batch 1 by 0.016 -- was measuring those two things,
not batch size. Under an epoch budget the effect disappears completely: across batch 1 to
64 the spread within each architecture is 0.0009-0.0021, inside the noise.

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

**A wrong turn worth recording:** an early probe concluded that linear scaling was wrong
for this model, after extrapolating linearly from batch 1's historical lr 0.01 to 0.32 at
batch 32 and watching it oscillate (0.9344 at 0.16, ~0.47 at 0.32). The rule was right;
the *anchor* was wrong. Batch 1 wants ~0.00125 under this schedule, not 0.01. Four
separate grids in this project bracketed an optimum from above only before this was
caught -- when probing an LR, bracket from both sides.

---

## Decision 4 -- warmup = 6400 / batch

At a fixed epoch budget this holds `max_steps * batch` constant, so warmup covers the
same **204,800 tokens** *and* the schedule reaches the same depth
(`(W/max_steps)**0.5 = 0.178`) at every batch size.

A fixed step count does neither. With 200 steps everywhere, depth varies **elevenfold**
across batch 1 to 128 (0.031 / 0.089 / 0.178 / 0.252 / 0.356) -- and depth moves the LR
optimum, which is exactly why the 3-epoch and 10-epoch optima differ. Any batch-size
comparison at fixed warmup steps is partly a depth comparison.

Ramp length itself barely matters. `--lr-decay-start` was added so the ramp could be
varied with the decay curve held fixed (in `(W/s)**alpha` the warmup length also sets the
height of the whole decay, which made every earlier warmup comparison uninterpretable --
a short ramp "won" twice purely by decaying faster). With `t0` pinned:

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

An earlier experiment had varied the cooldown at 3 epochs and found 0.2 best; that did
not transfer, because at 3 epochs a long cooldown eats real learning (the curve is still
climbing at +0.0059/epoch where it would start). Check what regime a schedule knob is
being tested in.

---

## Decision 6 -- batch size is a speed knob only

At a token-matched schedule (warmup `6400/batch`, so depth and warmup tokens identical),
3 epochs:

| batch | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| best accuracy | 0.9281 | 0.9371 | 0.9409 | **0.9413** | 0.9341 |
| optimal lr | <0.005 | 0.01 | 0.04 | 0.08 | 0.08 |
| throughput | ~3,000 tok/s | 6,400 | 6,050 | 7,000 | 5,900 |

Throughput saturates at batch 8 and is flat to 128; only batch 1 is penalised, at roughly
half. Batch 32 and 64 are tied within noise and 128 falls off. In the v2 sweep, with LR
scaled, accuracy across batch 1 to 64 is flat within each architecture.

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

---

# Part II -- the LM coded output head (full vocabulary)

Everything above is POS (18 classes). This part is the **LM** task: wiki2 word-level,
`V = 14,398`, same 6-layer body (`n_t 64 / n_c 8 / ctx 32 / emb 64 / h4 / ws8 / new-MHA`,
47,972,352 body table entries), batch 32, seed 42. The question is whether the output head
can be replaced by `D` coordinates plus a `(V, D)` code matrix -- `logits = y @ C.T + b` --
instead of one table column per class.

Measured across six sweeps, ~280 GPU-h: `LM_torch_output_code_E10` (does it work at all),
`LM_torch_schedule_probe` (budget and LR shape), `LM_torch_code_width` (how wide must `D` be,
and does a frozen code ever catch up), `LM_torch_lr_budget` (peak LR x budget, two rounds),
`LM_torch_depth` (where the freed parameters should go, plus a deeper ladder to L60).

## The settled LM configuration

```
--output-code learned --output-code-bias
--output-code-dim 112               # settled at L6 (Decision L3); the depth runs used 448,
                                    # and the deep cell has not been repeated at 112 (Open L7)
--num-layers 32                     # or 48 -- see Decision L5. n_c stays 8
--epochs 30                         # NOT more at depth: L24 at E=60 is worse than at E=30
--lr-scheduler cosine --cosine-min-lr-ratio 0.01
--lr-cooldown-frac 0.2              # applied in every Part II run (see note below)
--warmup-frac 0.0095
--max-learning-rate 0.00125          # x batch 32 = 0.04 effective, coded head, E>=30
--lr-batch-scaling linear --lr-reference-batch-size 1
--validation-interval-epochs 0.1
```

**Note (found 2026-10-14):** the Part II scripts kept the POS template's fixed flags
(`--lr-scheduler inverse-sqrt --lr-cooldown-frac 0.2 --warmup-steps 200 --epochs 10`) and
appended the arm's cosine flags after them. argparse keeps the last value, so cosine,
warmup-frac and the epoch count took effect. `--lr-cooldown-frac` was never overridden, so
**every Part II run is cosine to 1% multiplied by a linear ramp over the last 20%**. The
trainers print `lr_cooldown_frac = 0.2`. The block above now lists it; earlier copies
omitted it. New scripts must pass each flag once (`run_files/torch_LM_wt103_gpu.sh` does).

Best measured: **`L60_nc8_lr04` (D448) at 107.46 perplexity** (min 106.88; acc 0.2631,
no-unk ppl 133.1), 493.5M parameters, 26.2 h. The recommended point is **`L32` at 113.53**
(269.7M, 14.1 h) -- the deepest cell still smaller than the dense baseline -- or `L48` at
109.20 (397.6M, 21.6 h) if parameters are not the constraint.

Best dense baseline: **`dense_E60_lr04` at 120.65** (283.9M, 8.8 h).

**The headline is "deeper coded models beat the dense head at equal parameters", not "the
coded head is smaller at equal wall clock".** The earlier wall-clock claim (126.52 coded at
4.8 h vs 128.39 dense at 4.1 h) did not survive giving the dense head 60 epochs: at 120.65 it
beats every 6-layer coded run, including `D112_E120_lr04` (125.38 in 10.8 h). A 6-layer coded
head is smaller but worse. What holds up, all single-seed:

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
random code's minimum distance is far larger. This kills the "widen `D` to buy distance"
theory that `lut_torch/AGENTS.md` previously recommended -- see Decision L3.

## Decision L2 -- 30 epochs with a cosine anneal; `inverse-sqrt` is the wrong shape here

Unlike POS, the LM task is **not** saturated at 10 epochs -- it is budget-starved. All at
`D = 28` learned + bias:

| arm | schedule | epochs run | ppl | ppl @ ep 23.5 |
|---|---|---|---|---|
| (E10 reference) | inverse-sqrt, cooldown 0.2 | 10 | 187.7 | -- |
| `A1_E30_depthfree` | inverse-sqrt, warmup left at 200 steps | **23.5, preempted** | (167.3) | 167.3 |
| `A2_E30_depthpin` | inverse-sqrt, `--warmup-frac 0.0095` | 30 | 152.6 | 157.4 |
| `C3_E30_cooldown01` | inverse-sqrt, cooldown 0.1 | 30 | 151.9 | 157.1 |
| `C1_E30_alpha025` | inverse-sqrt `alpha 0.25`, depth-pinned | **23.5, preempted** | (150.1) | 150.1 |
| **`C2_E30_cosine`** | **cosine to 1%, depth-pinned** | **30** | **144.0** | **145.7** |
| `A3_E60_depthpin` | inverse-sqrt, depth-pinned | 60 | 138.0 | 150.8 |

**`A1` and `C1` were preempted at epoch 23.5** and never annealed; their finals are
mid-run values, so the last column compares every arm at that common epoch instead.

Three things, in order of size:

- **Cosine beats every `inverse-sqrt` variant at the same budget**, by 5.6% at 30 epochs
  (144.0 vs 152.6) and by 3.0% at the common epoch 23.5 (145.7 vs 150.1 for the best
  `inverse-sqrt` arm). This is the opposite of the POS finding in Decision 5, where the
  cooldown's shape was a measured null. The regimes differ: POS overfits and the anneal only
  collects a gap that is already there, while the LM task is still descending at the end of
  the budget, so the whole decay curve does real learning and its shape matters.
  `alpha 0.25` (`C1`) is the runner-up and beats `alpha 0.5` by 4.7% at epoch 23.5 --
  consistent with cosine's advantage being "hold a higher LR for longer, then anneal hard".
- **`A3`'s 60-epoch win is budget, not shape.** At epoch 30 `A3` is at 146.6, *behind*
  `C2`'s 144.0 at the same point; it only passes it by spending twice the compute. Cosine at
  `E = 60` has never been run and is the obvious next cell -- see Open L3.
- **Warmup must be pinned as a fraction, not a step count.** `--warmup-frac 0.0095`
  reproduces the 200-step depth of the 10-epoch schedule at 30 epochs; leaving warmup at 200
  absolute steps costs 6.3% at matched epoch (167.3 vs 157.4). Same mechanism as POS
  Decision 4 -- in `(W/s)**alpha` the warmup length sets the height of the whole decay -- but
  here it is worth 10 perplexity rather than nothing.
- **Budget is still the biggest single knob** and has not been exhausted: doubling to 60
  epochs buys another 9.6% even on the inferior `inverse-sqrt` shape. 30 is where this work
  stopped, not where the curve flattens.

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

**`binary` never saturates because it never fits.** Its gap stays at roughly zero across
the entire range -- it is still underfitting at `D = 1000`. It is not a regulariser; the
frozen code is a capacity and optimisation bottleneck. Concretely: **learned `D = 14`
(157.3) beats binary `D = 1000` (181.8)**, a >70x width disadvantage with no crossover
anywhere in the range.

Two things that make `binary` look better than it is:

- **It saves no storage at equal `D`.** The frozen `C` is a buffer so it does not appear in
  the printed parameter count, but it still occupies `V*D` floats. Including it, every
  binary arm's footprint equals the learned arm's exactly (e.g. 30,796,398 at `D = 1000`).
  The only real saving is gradient and optimiser state for `C`, at a cost of ~50 perplexity.
- **`B_D1000` has the highest raw accuracy of any arm** (0.2456, above `L_D1000`'s 0.2447)
  while being 38% worse in perplexity. On `acc_no_unk` the order flips back (0.2268 vs
  0.2310). The raw-accuracy edge is the bias term capturing `<unk>`, not the frozen code
  being competitive. This is the rank-`D` bottleneck behaving exactly as
  `lut_torch/AGENTS.md` warns: argmax survives, calibration does not.

Do not spend further GPU time on `binary`, and in particular **do not fix `ecoc_masks`
first and re-test**. Its minimum Hamming distance is stuck at 2 for every `D` from 28 to
1000 (a greedy-construction defect, not a property of the vocabulary), but the frozen
random `D = 28` control in Decision L1 shows distance is not the binding constraint.

## Decision L4 -- peak LR 0.04 at E>=30 for both heads; at L6 the dense head wins

`LM_torch_lr_budget`, all cosine to 1%, `--warmup-frac 0.0095`, seed 42. `D = 112` learned
+ bias. The `E=30 / 0.04` cell is `L_D112` from the width sweep (identical command line
apart from `--epochs`). Round 2 (`20260918-*`) added `D112_E120_lr04` and the two dense
`E=60` arms. Final-step perplexity:

| `D = 112` | peak 0.04 | peak 0.08 |
|---|---|---|
| E = 30 | 133.38 | 142.38 |
| E = 60 | 126.52 | 137.20 |
| E = 120 | **125.38** (min 124.57 @ ep 114) | 132.59 |

| dense (`V` logits) | peak 0.04 | peak 0.08 |
|---|---|---|
| E = 30 | 132.29 | **128.39** |
| E = 60 | **120.65** (min 119.93 @ ep 59) | *crashed at ep 37.5* |

**Round 2 overturns this decision's original headline.** Round 1 concluded that the coded
head reaches 126.52 in 4.8 h against the dense head's 128.39 in 4.1 h, and so was "68x
smaller at matched wall clock". Given 60 epochs, the dense head reaches **120.65 in 8.8 h**,
which beats every `L6` coded run, including `D112_E120_lr04` (125.38 in 10.8 h). At `L6`
the coded head is smaller but worse at any matched budget. Its advantage comes from spending
the freed parameters on depth (Decision L5), not from the head alone.

**The coded head's budget is exhausted near E=60.** E=60 -> E=120 at 0.04 is 126.52 ->
125.38, 0.9% for 2x compute, against the ~3.4% the previous revision extrapolated (it
predicted ~122). The train/val gap reached +0.64. More epochs at `L6` / `D = 112` are not
worth running.

**The dense head's LR preference also reverses with budget, tentatively.** At E=30 it
prefers 0.08; at E=60, 0.04 leads 0.08 at the last common point (123.90 against 125.23 at
epoch 37.5, same cosine schedule, so the same LR fraction). `dense_E60_lr08` died at its
final checkpoint save because its `/cs/snapless/.../LM_torch_lr_budget/<run>` directory did
not exist. That is an infrastructure fault, not divergence. Its 37.5-epoch value is not a
result.

**The `E=10` LR probe did not transfer, and it reversed.** At 10 epochs, 0.08 beat 0.04 by
8.4%; at 30 epochs it *loses* by 6.7% and at 60 by 8.4%. This is the same budget-depth
mechanism as POS Decision 3 -- a longer run anneals deeper on its own, so it wants a lower
peak -- but pointing the other way, and it invalidates the prediction in this file's
previous revision that the width sweep was understated by being run at 0.04. It was not;
0.04 was right. Nothing here diverged, including 120 epochs at 0.08.

**At E=30 the two heads want opposite LRs.** Coded prefers 0.04 and dense prefers 0.08
(128.39 against 132.29). Had the dense control been run at one LR only -- the POS
`new-MHA=off` mistake -- the comparison would have been wrong whichever LR was chosen. At
E=60 the dense preference appears to move to 0.04 too (above), the same reversal the coded
head showed between E=10 and E=30. Re-bracket LR whenever the budget changes.

**At 0.08 the coded model converges to ~132 regardless of budget** (137.20 at E=60, 132.59
at E=120, flat over its last 10 epochs: 132.55 -> 132.59). At 0.04 it goes lower but
saturates by E=120 too (above).

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
| L18 | *(died at 7.9 ep)* | 123.7 | 130.4 | 134.6 |
| L24 | **113.9** | 119.0 | 127.6 | 129.5 |
| dense L6 | 131.7 | -- | 127.9 | -- |

**Depth is monotone in all four ladders.** The deepest cell wins every column. Four
independent ladders agreeing matters more than the size of any one gap.

**The deeper ladder (`20260918-103308_*`, `run_files/torch_LM_deeper_gpu.sh`)** extended the
winning column (`n_c=8 / 0.04`, same flags apart from `--num-layers`) and refilled `L18`. Min
val perplexity, as in the table above (finals in brackets), with the train/val gap computed
from the final val loss:

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
of line with its neighbours. It is probably seed noise, which is unmeasured (Open L4), not a
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

**The coded head wins outright once depth is added, but by less than first measured.**
Against the E=30 dense head (127.9) the coded `L24` was 11% better at 28% fewer parameters.
The fair baseline is now `dense_E60_lr04` at 120.65 (Decision L4), the best dense run
measured. Against it:

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
and it should be repeated at 112 before `D` and depth are quoted together. The original
`L18 / n_c=8 / 0.04` was killed at epoch 7.9 on the `killable` partition. The deeper ladder
refilled it (`20260918-103308_L18_nc8_lr04_seed42`), so the stub run is superseded.
Single seed throughout.

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

**The best no-unk perplexity is 96.9 (`d256_L12_h8_do0.3`).** It reads high, but most of
the number comes from how and on what we score, not from the model:

- **Short context, every position scored.** Per-position loss of that checkpoint on the
  same 1,000 windows: ppl 202 at position 0 (one word of context), 155 at position 1, 114
  at position 2, and ~67-80 from position 8 on. Positions >= 16 alone give **ppl 72.5 /
  no-unk 86.2**, so the 32-word window with every position averaged costs ~11% on its
  own. The single-position values are noisy (1,000 tokens each); the >= 16 pool is not.
- **No-unk is a harder metric than standard WT2 perplexity.** `<unk>` is 8.8% of val
  targets and is easy to predict (ppl 13.2 on those targets), so removing it raises the
  number. Published WikiText-2 figures include `<unk>` targets, use a 33k vocabulary, and
  carry state across segments (AWD-LSTM is ~68 val with all of that). The closest
  comparable number here is the with-unk 81.4, or 72.5 with context. These are not
  apples to apples (smaller vocab, short window), so treat that as a sanity range, not a
  benchmark.

**"The data is not good enough": partly yes, as quantity, not as quality.** 2.16M training
tokens is what limits the larger models, and the sweep shows it. Every large or
lightly-regularised cell overfits: d512/L12/do0.1 bottoms at 97.0 at epoch 11.8 and ends
at 130.1, and d512/L6/do0.1 peaks at epoch 20.6. The best cells are the high-dropout ones,
and they are still improving at epoch 30. That is a data-size ceiling, consistent with
`L24_E60` memorising in Decision L5. Nothing here points to a *quality* problem with the
data: the tokens are verified identical to the LUT stream, and the per-position curve has
the shape expected of clean text. A larger corpus (WikiText-103, same pipeline) is the
test that would separate "small data" from "our models", for both model families.

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

- **L8 -- seed the GPT baseline (Decision L6).** All 12 cells are done. Seeds 43/44 for the best cell and for d64/L6 (array indices +12, +24).
  The best cells were still improving at epoch 30, so the baseline is probably under-, not
  over-, estimated. Also try dropout 0.4-0.5 at d256-d512. WikiText-103 through the same
  pipeline is the test for "data-limited".

- **L1 -- RESOLVED (Decision L4).** 0.04 is correct for the coded head at E >= 30; the
  `E=10` probe's preference for 0.08 reversed with budget. The dense head wants 0.08 at
  E=30 and, tentatively, 0.04 at E=60.
- **L2 -- RESOLVED (Decision L4).** Dense baselines exist at E=30 (128.39) and E=60
  (120.65). The "68x smaller at matched wall clock" claim is retracted. At `L6` the dense
  head wins; the coded head wins only with the freed parameters spent on depth.
- **L3 -- RESOLVED (Decision L4).** `D112_E120_lr04` = 125.38, not the ~122 predicted: the
  `L6` coded budget saturates near E=60. `dense_E60_lr04` = 120.65. **Still missing:
  `dense_E60_lr08`**, which died at its final save because its snapless checkpoint
  directory was missing (`sbatch_outputs/LM_TUNE2-*`). Rerun it after recreating the
  directory, or accept 0.04 on the epoch-37.5 lead.
- **L6 -- depth: flattening but not saturated (Decision L5).** L60 = 107.46 is the best
  cell, but the last 12 layers bought only 1.7% for +24% wall clock. No lower-layer
  starvation is visible through L60 (gap rises only +0.41 -> +0.46). Further depth is
  low-value. **The open questions at depth are now budget and regularisation:** `L24_E60`
  lost to `L24_E30` through memorisation (gap +0.77), so try E=15-20 and/or higher dropout
  at L32/L48 rather than more epochs or more layers.
- **L7 -- the depth grid ran at `D = 448`, Decision L3 settled on `D = 112`.** Repeat `L32`
  (and ideally `L48`) at 112 before quoting `D` and depth together. A narrower head may
  also overfit less, which is now the binding constraint.
- **L4 -- everything is seed 42, single seed.** The *repeat* floor is now ~0.11%
  (Decision L5, two byte-identical dense runs 0.14 ppl apart), but SEED variance remains
  unmeasured and the POS floors in Part I do not transfer to a different task and metric. The
  `D = 112` vs `D = 224` difference (133.4 vs 133.1) is certainly inside it, which is why
  `D = 112` is chosen over `D = 224` -- it is the smaller of two arms that are tied, not a
  measured winner. The `L24 -> L32` step (-0.6%, against ~4% for its neighbours) and the
  L18-vs-dense-E60 tie also need seeds before they mean anything. Two extra seeds for L24,
  L32, L48 and dense E60 is the cheapest way to know.
- **L5 -- `learnedD64_bias` (`LM_torch_output_code_E10`) and `R2_randomD64`
  (`LM_torch_schedule_probe`) never produced a `loss.csv`.** The `random_code_matrix`
  `n_bits >= 63` int64 overflow that killed them is fixed; the runs were not repeated.


---

## Still open

- **`new-MHA=off` LR is unbracketed.** `expD` ran 0.01/0.02/0.04/0.08 at batch 32 and was
  monotone increasing. Extend upward (0.08/0.16/0.32) before quoting the on-vs-off gap.
- **The 10-epoch LR rule (`0.0025 * batch`) is confirmed at batch 32 only.** Check batch
  8 (0.01 vs 0.02) and batch 64 (0.08 vs 0.16), ~4 GPU-h.
- **`E=8 + lr 0.08` has never been run together** with the chosen shape, though each half
  is validated. One confirming run before building on it.
- **Three cells cannot run**: `h8/mha=off/ws=full` is 1.68B parameters = 6.24 GiB of
  weights plus 6.24 GiB of dense gradient before activations, and the cluster mixes
  10.57 GiB to ~24 GiB cards behind one `--gres`. Pin the GPU by memory, or use
  `--table-grad sparse --cuda-graphs off`. Four cells of the *first* sweep failed the
  same way and were recorded only as "incomplete".
- **Everything is single-seed.** Differences under ~0.003 need 3+ seeds to mean anything.

---

# Part III -- WikiText-103 with GPT-2 BPE (decided 2026-10-14, runs pending)

Decision L6 left two questions: is the WT2 LM data-limited, and is the tokenizer part of
the problem? This part records the decisions made before running anything, so the
reasoning survives whatever the numbers turn out to be.

## Decision W1 -- the tokenizer was not the problem, but the LM line leaves it anyway

Measured on the WT2 pipeline (`d256_L12_h8_do0.3`, the 1,000 shared windows):

- The source file is WikiText-2 *v1*, which is already tokenized (33,277 types, rare
  words already `<unk>`, 2.5% of tokens). `word_tokenizer.py` tokenizes it again with
  `<unk>|<eol>|\w+|[^\w\s]` and `min_freq 10`. That more than doubles `<unk>` (6.8% of
  train, 8.8% of val targets), splits `@-@` / `@,@` / `@.@` into three tokens, and splits
  `'s` into two.
- Those artifacts are *easy* targets: `<unk>` ppl 13.2, `@` 2.8, `=` 1.4, `<eol>` 3.3,
  together 13.4% of targets. On real words + punctuation (86.6%) the same model is at
  **ppl 117.9**, against 81.4 overall. So the tokenizer flatters the headline. It does
  not explain it, and it cannot explain the GPT/LUT gap, because both see the identical
  stream.

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
memory-maps it after validating the manifest, and the text path is unchanged. Chosen over
encoding at startup (A: ~1.5 min and ~6 GB per job, and the token stream would depend on
whichever tiktoken a node has) and over an automatic cache (C: 12 array jobs starting
together race to write it). The run scripts copy both manifests into `config.txt`, so
every run records the exact token stream it trained on.

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

---

## Run index

| directory | what it is |
|---|---|
| `results/POS_torch_gpu_sweep/` | first sweep, 24 cells, fixed 200k steps, lr 0.01, no cooldown. 4 cells never ran (OOM) |
| `results/POS_torch_lr_probe/`, `lr_probe2/` | first LR probes; stage 1 used base-lr x scaling, stage 2 switched to absolute LR |
| `results/POS_torch_expA/`, `expA2/` | peak LR x budget at batch 32, and closing that grid |
| `results/POS_torch_expB/` | schedule shape: ramp, decay start, cooldown fraction |
| `results/POS_torch_expC/` | batch size at a token-matched schedule |
| `results/POS_torch_expD/` | does the LR rule transfer to `new-MHA=off` (4 cells OOM) |
| `results/POS_torch_expE/` | cooldown shape vs budget, 2 seeds, paired |
| `results/POS_torch_gpu_sweep_v2_E10/` | the rerun under this recipe, 32 cells |
| `results/LM_torch_output_code_E10/` | LM, does the coded head work: dense baseline, learned D14/D28/D64, binary D14. `learnedD64_bias` has no `loss.csv` |
| `results/LM_torch_schedule_probe/` | LM, budget and LR shape at D28: `A*` budget/warmup, `B*` peak LR, `C*` anneal shape, `R*` frozen-random controls. `R2_randomD64` never ran; `A1` and `C1` preempted at epoch 23.5 |
| `results/LM_torch_code_width/` | LM, `D` in {14,56,112,224,448,1000} x {learned, binary} at the C2 schedule, 30 epochs. Learned `D28` is absent -- it is `C2_E30_cosine` in the schedule probe |
| `results/LM_torch_depth/` | LM, layer count x `n_c` x peak LR at learned D448, 30 epochs, 18 cells, plus the deeper ladder (`20260918-103308_*`: L18 refill, L32, L48, L60, `L24_E60`). Decision L5. The `20260917-205223_L18_nc8_lr04` stub (killed at epoch 7.9) is superseded. Directories without `loss.csv` are aborted submissions |
| `results/LM_torch_lr_budget/` | LM, peak LR x budget at D112 plus dense controls at E=30 and E=60 (round 2: `20260918-*`). Decision L4. `dense_E60_lr08` crashed at epoch 37.5 on a missing checkpoint dir |
| `results/GPT_LM_baseline/` | LM, nanoGPT baseline (`gpt_baseline_train.py`): width {64..512} x depth {6,12} x dropout {0.1,0.3}, seed 42, same windows as the LUT LM runs. Decision L6. Arm label is in the directory name and `config.txt` (`arm=`) |

Each run's `config.txt` records its swept axis under `swept_*` keys and the full command;
trust those over directory names. Preempted runs say `DUE TO PREEMPTION` in the sbatch
output -- check before assuming a crash.
