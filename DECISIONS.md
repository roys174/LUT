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

Measured across three sweeps, ~45 GPU-h: `LM_torch_output_code_E10` (does it work at all),
`LM_torch_schedule_probe` (budget and LR shape), `LM_torch_code_width` (how wide must `D` be,
and does a frozen code ever catch up).

## The settled LM configuration

```
--output-code-dim 112 --output-code learned --output-code-bias
--epochs 60                         # 30 is the cheap point; 60 is 5.1% better
--lr-scheduler cosine --cosine-min-lr-ratio 0.01
--warmup-frac 0.0095
--max-learning-rate 0.00125          # x batch 32 = 0.04 effective. CONFIRMED at E>=30
                                    # for the CODED head; the dense head wants 0.0025
--lr-batch-scaling linear --lr-reference-batch-size 1
--validation-interval-epochs 0.1
```

Best measured: **`D112_E60_lr04` at 126.52 perplexity** (val loss 4.8404, acc 0.2467,
no-unk ppl 157.7) in 4.79 h, with a **3,461,982-parameter head -- 68x smaller** than the
dense 235,896,832-entry head, which is 83% of the uncoded model.

**The compression claim is "68x smaller at equal wall clock", not "smaller and better at
equal epochs".** At 30 epochs with each head at its own best LR the dense head wins
(128.39 against 133.38, 3.9%). The coded head is 1.7x faster per epoch (0.080 h/epoch
against 0.137), so at ~4.5 h of GPU it reaches 126.52 against the dense head's 128.39 --
see Decision L4.

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

## Decision L4 -- peak LR 0.04 for the coded head, 0.08 for the dense one; budget beats both

`LM_torch_lr_budget`, all cosine to 1%, `--warmup-frac 0.0095`, seed 42. `D = 112` learned
+ bias. The `E=30 / 0.04` cell is `L_D112` from the width sweep (identical command line
apart from `--epochs`).

| `D = 112` | peak 0.04 | peak 0.08 |
|---|---|---|
| E = 30 | 133.38 | 142.38 |
| E = 60 | **126.52** | 137.20 |
| E = 120 | *not run* | 132.59 |

| dense (`V` logits) | peak 0.04 | peak 0.08 |
|---|---|---|
| E = 30 | 132.29 | **128.39** |

**The `E=10` LR probe did not transfer, and it reversed.** At 10 epochs, 0.08 beat 0.04 by
8.4%; at 30 epochs it *loses* by 6.7% and at 60 by 8.4%. This is the same budget-depth
mechanism as POS Decision 3 -- a longer run anneals deeper on its own, so it wants a lower
peak -- but pointing the other way, and it invalidates the prediction in this file's
previous revision that the width sweep was understated by being run at 0.04. It was not;
0.04 was right. Nothing here diverged, including 120 epochs at 0.08.

**The two heads want different LRs, in opposite directions.** Coded prefers 0.04, dense
prefers 0.08 (128.39 against 132.29). Had the dense control been run at one LR only -- the
POS `new-MHA=off` mistake -- the comparison would have been wrong whichever LR was chosen.

**`E=120` underperformed because it was only run at 0.08, not because the budget stopped
paying.** At 0.08 the model converges to ~132 regardless of budget (137.20 at E=60,
132.59 at E=120, and flat over its last 10 epochs: 132.55 -> 132.59). At 0.04, E=60 alone
reaches 126.52 and was still improving. `E=120 / 0.04` is unrun and is the best remaining
cell; scaling E=60->E=120's 3.4% gain onto it puts it near 122.

**Overfitting is now visible and is the coming constraint.** Train/val gap at `D = 112`:
+0.217 at E=30, +0.383 at E=60, +0.583 at E=120 -- and `E=60/0.04` peaked at epoch 51.5 of
60. The dense head's gaps are far smaller (+0.07 at 0.04, +0.15 at 0.08). So the next
budget increase probably wants more dropout or more data alongside it, not epochs alone.

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

**Depth is monotone in all four ladders, with no flattening at 24.** The deepest cell wins
every column. Four independent ladders agreeing matters more than the size of any one gap.

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

**The coded head now wins outright, not just on size.** Best coded cell 113.9 at 205.7M
entries against best dense 127.9 at 283.9M: **11% better perplexity at 28% fewer
parameters.** It also beats `LM_torch_lr_budget`'s `D112 / E=60 / 0.04` (125.5) at half the
budget, so depth at `E = 30` is worth more than 2x epochs at `L6`.

**Decision L4's LR split replicates independently.** Coded cells prefer 0.04 in all eight
pairs; the dense baseline prefers 0.08 (127.9 against 131.7). Two experiments, two head
types, same answer in both directions.

**First LM replicate floor: ~0.11%.** `base_L6_lr08` here and `dense_E30_lr08` in
`LM_torch_lr_budget` are byte-identical command lines at seed 42 -- verified flag by flag --
and landed on 127.87 and 128.01, 0.14 ppl apart. CUDA's `index_add_` atomics are not
deterministic, so that is the run-to-run floor for a *repeat*, not for a seed change; seed
variance is still unmeasured and will be larger. It does mean the `n_c` and depth effects
here (4-16%) are two orders of magnitude clear of noise, while the `L24 n_c=8` vs
`L24 n_c=10` accuracy tie (0.2548 vs 0.2554) is not.

**Caveats.** Run at `D = 448`, not the `D = 112` of Decision L3 -- chosen before L3 settled,
and it should be repeated at 112 before `D` and depth are quoted together. `L18 / n_c=8 /
0.04` was killed at epoch 7.9 on the `killable` partition and is the one hole in the grid,
in the winning family. Single seed throughout.

## Still open (LM)

- **L1 -- RESOLVED (Decision L4).** 0.04 is correct for the coded head at E >= 30; the
  `E=10` probe's preference for 0.08 reversed with budget. The dense head wants 0.08.
- **L2 -- RESOLVED (Decision L4).** The 30-epoch dense baseline exists: 128.39 at its best
  LR. The compression claim is now "68x smaller at matched wall clock", not "smaller and
  better at matched epochs".
- **L3 -- `E = 120` at peak 0.04 is the highest-value unrun cell**, ~9.6 h, expected near
  122. The dense head at `E = 60` is the other one: it would say whether the coded head's
  wall-clock win at ~4.5 h survives giving the dense head the same wall clock (~8.2 h).
  Run both before quoting any head-vs-head result. **Superseded in part by Decision L5**:
  the coded head already beats the dense one at matched epochs once the freed budget is
  spent on depth, so the head-vs-head claim no longer rests on wall clock.
- **L6 -- depth has not saturated.** `L24` is the deepest cell run and wins every ladder,
  with `L12 -> L24` still buying 9.3 ppl at 0.04 / `n_c=8`. `L32` and `L48` are the
  highest-value unrun cells. Note the stack carries 2 unnormalised residual adds per layer
  and no LayerNorm, and the surrogate decays as `1/(1+|u|)^2`, so a deep stack could in
  principle starve its lower layers -- 24 layers shows no sign of it, but log mean `|u|`
  per layer if a deeper ladder flattens, before concluding depth stopped paying.
- **L7 -- the depth grid ran at `D = 448`, Decision L3 settled on `D = 112`.** Repeat the
  best depth cell at 112 before quoting `D` and depth together.
- **L4 -- everything is seed 42, single seed.** The *repeat* floor is now ~0.11%
  (Decision L5, two byte-identical dense runs 0.14 ppl apart), but SEED variance remains
  unmeasured and the POS floors in Part I do not transfer to a different task and metric. The
  `D = 112` vs `D = 224` difference (133.4 vs 133.1) is certainly inside it, which is why
  `D = 112` is chosen over `D = 224` -- it is the smaller of two arms that are tied, not a
  measured winner.
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
| `results/LM_torch_depth/` | LM, layer count x `n_c` x peak LR at learned D448, 30 epochs, 18 cells. Decision L5. `L18_nc8_lr04` killed at epoch 7.9 |
| `results/LM_torch_lr_budget/` | LM, peak LR x budget at D112 plus the 30-epoch dense control. Decision L4 |

Each run's `config.txt` records its swept axis under `swept_*` keys and the full command;
trust those over directory names. Preempted runs say `DUE TO PREEMPTION` in the sbatch
output -- check before assuming a crash.
