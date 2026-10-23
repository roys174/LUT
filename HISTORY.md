# HISTORY.md

How this project got where it is: what was built, which experiments ran, which result
trees belong to which era, and the conclusions that were later overturned. It is the
process record, kept apart so `AGENTS.md` (how to work here) and `DECISIONS.md` (what is
currently settled, and the numbers behind it) stay short.

This describes what commits and saved `config.txt` files show was attempted. Do not infer
conclusions from run names alone; use loss files, stdout, W&B and the analysis tools. Run
directory and W&B names are occasionally stale or inconsistent — trust the command line in
`config.txt` (and its `swept_*` keys) over the name, and check for `DUE TO PREEMPTION` in
the sbatch output before reading a short run as a crash.

## Timeline

| when | era | main result trees |
|---|---|---|
| Feb–Mar 2026 | `main.py` built and vectorised: Numba kernels, soft attention, nanoGPT submodule | — |
| May 2026 | POS task on word tokens, old attention | `results/POS/` |
| Jun–Aug 2026 | new attention (Q/K/V/Final), update statistics, PE/QK variants, Final sharing, new MHA, XOR aggregation | `results/POS_new_attn/` |
| Sep 2026 | `lut_torch` port, GPU, schedule retune for POS | `results/POS_torch_gpu*/`, `POS_torch_lr_probe*/`, `POS_torch_exp*/` |
| Sep–Oct 2026 | LM with full vocabulary: coded output head, schedule, width, depth | `results/LM_torch_*/` |
| Oct 2026 | nanoGPT LM baseline; WikiText-103 with GPT-2 BPE | `results/GPT_LM_baseline/`, WT103 runs pending |

## `main.py` era (Feb–Aug 2026)

The code progressed roughly as:

1. Vectorize and Numba-compile LUT forward/backward paths.
2. Add soft attention, pre-norm LayerNorm, dropout, batching, and sparse gradient replay.
3. Add word tokenization, POS caching/inference/accuracy, W&B, save/load, and multi-seed scripts.
4. Instrument which LUT rows, coordinates, and row indices (`j`) are updated.
5. Introduce the current Q/K/V/Final new-attention mechanism and its backward/replay paths.
6. Run positional encoding, comparison-layout, QK normalization, Final sharing, MHA, and LR-schedule experiments.
7. Add the committed `--comparisons-per-bit` XOR aggregation architecture (commit `adce4de`).

### Baselines and POS pipeline

- The branch inherited old hard attention, soft attention, LayerNorm/dropout, fp16, factored output, and batch parallelism.
- POS work added word inputs, spaCy universal-POS targets, cache reuse, correct `<unk>` alignment, inference, accuracy, and a token-majority baseline.
- `results/POS/` contains old-attention single- and multi-seed runs. Multi-seed scripts use seeds 42-45.
- `run_nanoGPT_POS.sh`, `infer_gpt_pos.py`, and `pos_majority_baseline.py` provide transformer and empirical baselines.

### New attention

- Commits `7433e31`, `1775307`, and `6fd1cc0` progressed from initial forward work to separated Q/K/V/Final and complete backward/replay support.
- Follow-ups integrated new tables into model-size reporting and update stats.
- Common full POS shape: context 32, embedding 64, 6 layers, 4 heads, `n_t=64`, `n_c=6`, positional/QK dim 5, dropout 0.2, seed 42. Runs also varied 10 layers, 4/8/16 heads, and `n_t` 16/32/64/128.
- `results/POS_new_attn/20260629-231508` is a historical sanity reference: existing notes record validation loss about 0.398, perplexity 1.49, and accuracy 0.869 near step 99,000. It is not a guaranteed benchmark.

### Update coverage and row behavior

- `20260802-...stats...` runs varied `n_t` 16/32/64, random/overlap/no-overlap comparisons, direct-sign QKPE, PE-only, frozen PE, and frozen PE+PE-only.
- Instrumentation grew from row counts/histograms to `j` distributions, cumulative snapshots, per-coordinate counts, and KL divergence from uniform.
- Multiple-checkpoint work supports comparing initial/intermediate/final `S` rows.

### PE and QK variants

- Direct-sign QKPE, deterministic overlap/no-overlap layouts, PE-only, frozen PE, and sign-pattern PE initialization were tested.
- QK normalization was run at least in `results/POS_new_attn/20260806-102506`. Do not trust later run names that say “norm”; confirm `--normalize-qk` in `config.txt`.

### Final sharing, MHA, schedules, and XOR aggregation

- Final sharing was tried with `n_t=64, weight-sharing=8` and `n_t=128, weight-sharing=16`; later MHA runs use sharing 8 or 16.
- New MHA runs cover 4, 8, and 16 heads, usually with frozen PE, direct-sign QKPE, random comparisons, and Final sharing.
- Scheduler experiments varied warmup 1000/2000, cosine versus inverse sqrt, peak LR (notably 0.01), and inverse-sqrt exponent 0.32 versus 0.5.
- Later runs varied `n_c`/`attn_qk_n_c` through 6, 7, 8, and 9. Some W&B names say `WS_16` while the command says `--weight-sharing 8`; trust `config.txt`.
- XOR aggregation was implemented in `adce4de`. The focused comparison suite verifies exact legacy behavior at `--comparisons-per-bit 1`, while fan-in 2 smoke runs cover old hard, soft, and new attention.

## Torch port and POS schedule retune (Sep 2026)

- `lut_torch/` ported the new-attention / new-MHA architecture (`b387265`), then added
  CUDA graphs and stacked heads (`083d103`). Equivalence with `main.py` is in `AGENTS.md`
  ("`main.py` vs `lut_torch`"); speed history is in `lut_torch/PERFORMANCE.md`.
- The training schedule was retuned from scratch: budget in epochs rather than steps, peak
  LR scaled linearly with batch, warmup `6400/batch`, and a terminal cooldown. Outcome:
  `new-MHA=on` beats `new-MHA=off` by 0.010-0.017 with 3-7x fewer parameters, and reaches
  0.9454 in ~57 min where the old schedule reached 0.9445 in 3.30 h (and *ended* at 0.9407
  after 9.02 h). About 40 GPU-h of probes plus two full sweeps. DECISIONS.md Part I.
- Result trees, in order: `POS_torch_gpu_sweep/` (first sweep, fixed 200k steps),
  `POS_torch_lr_probe*/` and `POS_torch_exp{A,A2,B,C,D,E}/` (the tuning probes),
  `POS_torch_gpu_sweep_v2_E10/` (the rerun).

## LM with a full vocabulary (Sep–Oct 2026)

- Output head switched to a matmul with bias and a separate LR (`0266190`); full-vocab
  support with chunked LUT backward, ECOC/random output codes and the chat UI (`44a9237`).
- Six sweeps, ~280 GPU-h, in this order: `LM_torch_output_code_E10` (does a coded head work
  at all), `LM_torch_schedule_probe` (budget and LR shape), `LM_torch_code_width` (how wide
  must `D` be, does a frozen code catch up), `LM_torch_lr_budget` (peak LR x budget, two
  rounds), `LM_torch_depth` (where freed parameters go, plus a deeper ladder to L60).
  Scripts: `run_files/torch_LM_{output_code,schedule_probe,code_width,lr_budget,lr_budget2,depth,deeper}_gpu.sh`.
- nanoGPT LM baseline (`1f24997`, `results/GPT_LM_baseline/`) on the same windows.
- WikiText-103 with GPT-2 BPE (`7331b73`); decisions made before any run, DECISIONS.md
  Part III.

## Wrong turns and superseded claims

Each of these was believed for a while. They are recorded so nobody re-derives them.

### POS

- **Budgeting in steps.** The first GPU sweep (`POS_torch_gpu_sweep/`) found batch 32 beat
  batch 1 by 0.016. It was measuring 32x more passes over the data and 32x the parameter
  travel, not batch size; under an epoch budget the effect vanished (DECISIONS D1).
- **"Linear LR scaling is wrong for this model."** An early probe extrapolated linearly
  from batch 1's historical lr 0.01 to 0.32 at batch 32, watched it oscillate (0.9344 at
  0.16, ~0.47 at 0.32), and concluded the rule failed. The rule was right; the *anchor* was
  wrong — batch 1 wants ~0.00125 under this schedule, not 0.01. Four separate LR grids
  bracketed the optimum from above only before this was caught (DECISIONS D3).
- **Warmup comparisons before `--lr-decay-start`.** In `(W/s)**alpha` the warmup length
  also sets the height of the whole decay, so a short ramp "won" twice purely by decaying
  faster. Every warmup comparison before t0 could be pinned was uninterpretable (D4).
- **Cooldown length tuned at 3 epochs.** 0.2 looked best at 3 epochs and did not transfer:
  at 3 epochs a long cooldown eats real learning. Check what regime a knob is tested in (D5).
- **The port looked better than `main.py`.** It was the validation windows, not the model;
  same-window scoring gave identical accuracy (`AGENTS.md`, "Comparing Runs").

### LM

- **"The coded head is 68x smaller at matched wall clock."** Round 1 of
  `LM_torch_lr_budget` had coded D112 at 126.52 ppl in 4.8 h against dense at 128.39 in
  4.1 h. Round 2 gave the dense head 60 epochs and it reached 120.65 in 8.8 h, beating
  every 6-layer coded run including `D112_E120_lr04` (125.38 in 10.8 h). Retracted; the
  coded head only wins once its freed parameters go into depth (DECISIONS L4, L5).
- **"Doubling the budget to E=120 reaches ~122."** It reached 125.38: the coded L6 budget
  saturates near E=60.
- **"The width sweep is understated because it ran at 0.04."** Predicted from the E=10
  probe, where 0.08 beat 0.04 by 8.4%. At E=30 and E=60 the preference reversed; 0.04 was
  right.
- **"Widen `D` to buy Hamming distance for a frozen code."** Once recommended in
  `lut_torch/AGENTS.md`. Frozen random D=28 scores the same as frozen binary D=14 despite a
  far larger minimum distance: freezing costs, distance does not (DECISIONS L1).
- **The doubled cooldown flag (found 2026-10-14).** The Part II scripts kept the POS
  template's fixed flags (`--lr-scheduler inverse-sqrt --lr-cooldown-frac 0.2
  --warmup-steps 200 --epochs 10`) and appended each arm's cosine flags after them.
  argparse keeps the last value, so cosine, warmup-frac and the epoch count took effect,
  but `--lr-cooldown-frac` was never overridden: every Part II run is cosine to 1%
  multiplied by a linear ramp over the last 20%. Earlier copies of the settled config
  omitted it.
- **Last-position-only LM validation.** `main.py` scored only the last position of each
  window; `lut_torch` now scores all of them. LM numbers from before the change are not
  comparable with later ones.

### Infrastructure incidents

- `LM_torch_schedule_probe`: `A1_E30_depthfree` and `C1_E30_alpha025` were preempted at
  epoch 23.5 and never annealed; their finals are mid-run values.
- `learnedD64_bias` (`LM_torch_output_code_E10`) and `R2_randomD64`
  (`LM_torch_schedule_probe`) never produced a `loss.csv`: an `n_bits >= 63` int64 overflow
  in `random_code_matrix`, since fixed; not rerun.
- `dense_E60_lr08` (`LM_torch_lr_budget` round 2) died at its final checkpoint save
  because its snapless checkpoint directory did not exist; its epoch-37.5 value is not a
  result.
- `LM_torch_depth`: the original `L18 / n_c=8 / 0.04` (`20260917-205223_*`) was killed at
  epoch 7.9 on the `killable` partition; the deeper ladder refilled it
  (`20260918-103308_L18_nc8_lr04_seed42`). Directories without `loss.csv` are aborted
  submissions.
- POS first sweep: 4 cells never ran (OOM), recorded only as "incomplete"; `expD` lost 4
  cells the same way.
