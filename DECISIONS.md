# DECISIONS.md

Why the training configuration is what it is. `AGENTS.md` and `lut_torch/AGENTS.md` carry
the verdict; this file carries the measurements behind it, so that a later change can see
which numbers it would be overturning rather than rediscovering them.

Measured on the POS task (wiki2, 2,156,877 training positions, context 32) at
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

Each run's `config.txt` records its swept axis under `swept_*` keys and the full command;
trust those over directory names. Preempted runs say `DUE TO PREEMPTION` in the sbatch
output -- check before assuming a crash.
