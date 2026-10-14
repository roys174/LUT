# AGENTS.md

Read this before changing this directory, especially `main.py`. This is research code: preserve experiment provenance and prefer small, testable changes over broad cleanup.

## Repository Purpose and Current State

This is a NumPy/Numba lookup-table (LUT) language model derived from the SNN/LUT transformer idea. `main.py` is the original implementation and supports next-token language modeling and part-of-speech (POS) prediction. `lut_torch/` is a PyTorch port of the current architecture that reproduces `main.py` exactly (see "The `lut_torch/` PyTorch Port"). `nanoGPT_POS/` is the transformer POS baseline.

The active branch is ahead of `main` by a long experiment series. The progression was roughly:

1. Vectorize and Numba-compile LUT forward/backward paths.
2. Add soft attention, pre-norm LayerNorm, dropout, batching, and sparse gradient replay.
3. Add word tokenization, POS caching/inference/accuracy, W&B, save/load, and multi-seed scripts.
4. Instrument which LUT rows, coordinates, and row indices (`j`) are updated.
5. Introduce the current Q/K/V/Final new-attention mechanism and its backward/replay paths.
6. Run positional encoding, comparison-layout, QK normalization, Final sharing, MHA, and LR-schedule experiments.
7. Add the committed `--comparisons-per-bit` XOR aggregation architecture (commit `adce4de`).

Working-tree warning: `nanoGPT_POS/` is untracked user work. Do not discard, overwrite, or silently reformat it.

## Start Here

- `main.py`: model structures, LUT construction/indexing, Numba kernels, forward/backward, batching/replay, CLI, training, validation, checkpointing, update statistics, and W&B.
- `lut_torch/`: the PyTorch port. Start at `lut_torch/functional.py` (the only custom autograd) and `lut_torch/modules.py`.
- `word_tokenizer.py`: word vocabulary, tokenization, universal POS labels, spaCy alignment, and cached POS encoding.
- `run_files/run_POS_one.sh`: newest single-run template; uses new MHA, scheduler control, frozen PE, direct-sign QKPE, and the legacy-equivalent `--comparisons-per-bit 1`. Its `--weight-sharing` line is currently commented out.
- `run_files/torch_POS_run.sh`: the torch trainer (`python -m lut_torch.train`) at the same experiment shape. Note `run_files/` is gitignored, so these scripts are local-only.
- `run_files/torch_POS_output_code_gpu.sh`: GPU array job for the low-dimensional coded
  output head (`--output-code-dim`), three arms (uncoded baseline, learned code, frozen
  binary code) at the GPU sweep's best cell. Its header explains why the baseline must be
  rerun from that script and why validation loss, not just accuracy, decides the result.
- `run_files/torch_LM_output_code_gpu.sh`, `torch_LM_schedule_probe_gpu.sh`,
  `torch_LM_code_width_gpu.sh`, `torch_LM_depth_gpu.sh`, `torch_LM_lr_budget_gpu.sh`: the
  full-vocabulary (`V = 14,398`) coded-head experiments, in the order they were run. The
  settled configuration and every number behind it are in DECISIONS.md Part II; the short
  version is `--output-code-dim 112 --output-code learned --output-code-bias --epochs 30
  --lr-scheduler cosine --cosine-min-lr-ratio 0.01 --warmup-frac 0.0095`.
  **All five share a preamble that exists because of two cluster faults -- copy it into any
  new run script rather than writing a bare `source .venv/bin/activate`:**
    - *Per-node venv selection.* A venv does not contain a Python: `.venv/bin/python` is a
      symlink to `/usr/bin/python3`, so it runs **whatever version the node has**, and looks
      for packages in `lib/python<version>/site-packages`. `.venv` has only
      `lib/python3.11`, `.venv313` only `lib/python3.13`. The cluster is being re-imaged
      from 3.11 to 3.13 node by node, so on a 3.11 node `.venv` works and `.venv313` fails,
      and on a 3.13 node the reverse -- neither is wrong, the node decides. The scripts try
      each in turn and take the first whose interpreter can `import numpy, torch`, keep the
      winner in `PY`, and record it in `config.txt` as `venv=`. Do not use bare `python`
      after that point; use `"${PY}"`, so the provenance and the trainer cannot disagree.
    - *GPU preflight and `--exclude`.* The re-imaged nodes ship **no NVIDIA driver
      userspace** -- `nvidia-smi` is absent, so `libcuda.so` is missing and
      `torch.cuda.is_available()` is False. SLURM advertises their A10s and allocates them
      anyway (`gres/gpu:a10=1` in `sacct -o AllocTRES`), so a job lands, takes a GPU it
      cannot use, and dies inside argparse with a usage dump whose last line is the real
      error. Reproducer: `srun -w arion-01 --gres=gg:g0:1 --pty nvidia-smi` ->
      `No such file or directory`. The scripts carry
      `#SBATCH --exclude=arion-01,binky-01,creek-01,dumfries-004,drape-01,incitatus-01` and
      assert `torch.cuda.is_available()` right after picking the venv, failing with one
      legible line. Prune the exclude list as nodes are repaired; the preflight is what
      catches the ones not yet on it.
- `gpt_baseline_train.py` + `run_files/GPT_LM_baseline_gpu.sh`: nanoGPT LM baseline for the
  LUT LM. The trainer wraps `nanoGPT_POS/model.py`'s `GPT` but takes data, validation
  windows, `evaluate()` and the epoch/cosine schedule from `lut_torch`, so at the same
  `--seed`/`--testing-length` its `loss.csv` scores the same snippets as a `lut_torch` LM
  run (`test_gpt_baseline.py`). The optimizer is the transformer's own (AdamW). The script
  sweeps width x depth x dropout at seed 42; results go to `results/GPT_LM_baseline/`.
- `prepare_wikitext103.py` -> `nanoGPT_POS/data/wiki103/`: WikiText-103 raw, GPT-2 BPE
  `.bin` token files (full train, 2M/8M/32M-word article-aligned prefixes, val, test),
  each with a `.bin.json` manifest. Both trainers take a `.bin` wherever they take a text
  file, with `--tokenizer gpt2`. The val set is WikiText-2's, word for word (checked;
  recorded in `val_gpt2.bin.json`). Decisions W1-W4 in DECISIONS.md Part III.
  Runs: `run_files/GPT_LM_wt103_gpu.sh` (12-cell GPT data x size ladder) and
  `run_files/torch_LM_wt103_gpu.sh` (LUT; arm 0 = smoke, arms 1-4 = wave 2).
- `run_files/run_POS_multiseed_new_attn.sh`: seeds 42-45 in parallel and then `analyze_multiseed_metrics.py`.
- `run_files/for_srun_POS_new_attn.sh`: configurable short/statistics runs launched with `srun`.
- `run_files/run_POS_one_PE_only.sh`: PE-only ablation.
- `run_files/run_POS_one_multi_ckpt.sh`: multiple-checkpoint experiment; see caveat below.
- `run_files/run_POS_load_smoke.sh` and `run_files/run_POS_one_load_ckpt.sh`: checkpoint-load checks/templates; inspect paths before running.
- `run_files/run_nanoGPT_POS.sh`: PyTorch baseline. Inspect its working-directory assumptions before copying paths.
- `results/**/config.txt`: strongest local record of the exact command submitted for a run. Directory/run names are descriptive but occasionally stale or inconsistent; trust the command line inside the config.
- `loss.csv`: `step, epoch, tokens, loss, perplexity, accuracy, loss_no_unk, perplexity_no_unk, accuracy_no_unk, loss_per_word, perplexity_per_word` (11 columns; runs before 2026-10-14 have only the first 9). The `_no_unk` triple scores the same predictions with `<unk>` targets excluded; nothing is masked in training, both scorings are reported. It is word-tokenizer only and NaN under BPE. The `_per_word` pair is the exact per-word loss for BPE runs (DECISIONS.md W2) and the headline there; it is NaN for the word tokenizer. `perplexity` is exactly `exp(loss)` — but W&B's `train/perplexity` is `exp(EMA loss)` while `train/loss` is the raw last-minibatch mean, so those two panels rank runs differently.

## Where checkpoints live

**Checkpoints belong on `/cs/snapless`, not in `results/`.** `results/` is on the lab
filesystem, which is quota'd lab-wide (`/cse/labs/roys`, 9,216 GB for everyone), and a
single LUT checkpoint runs to tens of gigabytes -- one `main.py` `.npz` was 43 GB. When
that quota fills, running jobs stop being able to write `loss.csv` and `config.txt`, so
this is a correctness problem and not only a tidiness one.

The current run scripts already do the right thing: `CHECKPOINT_ROOT` points at
`/cs/snapless/roys/yuval.rubin/LUT_checkpoints` and each run records `checkpoint_dir=` in
its `config.txt`. **No code change is needed** -- `main.py` and `lut_torch` both write
wherever `--save-model` says, and the scripts say snapless.

Historical checkpoints (the `main.py`-era `.npz` files under `results/POS*` and the older
`results/POS_torch_gpu/*.pt`) were relocated to
`/cs/snapless/roys/yuval.rubin/LUT_checkpoints/legacy/<same path under results/>`, with a
**symlink left at every original path**. `infer_lut_pos.py`, the `analyze_*` / `compare_*`
tools and any `--load-model` path in an old `config.txt` therefore keep working unchanged.
Each affected run also has `checkpoint_moved_to=` appended to its `config.txt`, so the
location survives even if a symlink is broken or the tree is copied without following
links. Do not "clean up" those symlinks -- they are the checkpoints.

Related: `sbatch_outputs/*.txt` were 5.2 GB, of which 99.9% was tqdm carriage-return
animation frames (one 130 MB log held 900k `\r` against 1.1k `\n`). They are now pruned to
the last segment of each line -- exactly what the terminal displayed -- for 67 MB total.
New logs accumulate the same way; prune rather than gzip, which keeps them greppable and
is ~28x smaller than gzipping the frames.

## The `lut_torch/` PyTorch Port

`lut_torch/` reimplements the `--new-attention` / `--new-MHA` architecture on PyTorch so
that torch owns autograd, optimizers, schedulers and checkpointing. `main.py` is not
modified and remains the reference implementation that the port is measured against.

**`lut_torch/AGENTS.md` is the authoritative document for that codebase.** Read it before
changing anything under `lut_torch/`. It covers the design (why a discrete lookup fits
torch at all), the invariants that break silently (RNG draw order, loss scaling, sparse
table gradients), how CPUs are used and how to benchmark them, the flag set, the parity
contract, and the performance history. Do not duplicate that material here.

In brief, for orientation only:

- Only two gradients are custom: the table scatter-add (which is an embedding's backward,
  returned sparse) and the input surrogate (which is exactly `dphi/du` for
  `phi(u) = 0.5/(1+|u|)`). Both live in `lut_torch/functional.py`.
- The custom step is defined over *margins*, not the layer input, so QK normalization, PE
  routing and the Q/K/PE gradient scattering are ordinary torch that autograd derives.
- A real batch dimension replaces `BatchElement`, `SparseGradAccumulator`,
  `apply_averaged_gradients`, the `_replay_*` kernels and all four thread-pool flags.
- Checkpoints are torch `state_dict` `.pt` files and resume fully; they are not `.npz` and
  the `analyze_*` / `compare_*` / `infer_lut_pos.py` tools cannot read them.
- Not ported (still in `main.py`): old hard attention, `--soft-attention`,
  `--factored-output`, `--fp16`, `no-overlap`/`overlap` modes, `--halve-lr-at-20k-40k`,
  `--save-multiple-checkpoints`, sampling, and `LUTUpdateStats`.

Run scripts: `run_files/torch_POS_run.sh`. Tests: see "Validation Expectations" below.


## Cluster Sandbox and Command Execution — Read This Before Running Anything

**The default command/file sandbox is incompatible with this cluster. Do not repeatedly retry it or debug the repository when it fails.**

The characteristic infrastructure error is:

```text
bwrap: Creating new namespace failed, likely because the kernel does not support user namespaces.
bwrap must be installed setuid on such systems.
```

This happens before the requested command runs. It is not a Python, test, dependency, or repository failure.

Follow these rules:

1. Run terminal commands for this repository with the execution tool's host/escalated sandbox permission from the first attempt. Set the working directory explicitly to `/cs/labs/roys/yuval.rubin/LUT/LUT` and give a narrow, honest justification.
2. If a command was accidentally started in the default sandbox and returns the `bwrap` error above, rerun that same command once with host/escalated permission. Do not keep retrying the default sandbox, modify the command to evade it, or investigate application code.
3. Escalated execution only bypasses the broken namespace sandbox; it does not authorize destructive, unrelated, network, or system-wide actions. Keep commands scoped to this repository and request any additional approval normally.
4. The dedicated `apply_patch` helper currently enters the same broken sandbox and may fail before reading a file. Try the prescribed patch helper once when required. If it fails with the exact `bwrap` error, use `git apply` through an escalated terminal call with a small reviewed unified diff. Do not rewrite tracked files with `cat`, shell redirection, `sed -i`, or Python.
5. A long command may return a process/session ID instead of finishing. Poll that existing process with the execution tool's stdin/poll operation; do not launch a duplicate test or training run. Keep each wait under 60 seconds so the user continues receiving updates.
6. Preserve dirty and untracked user work. Check `git status --short` before editing, patch only the intended file, and never use reset/checkout/clean to recover from a patch problem.

Environment notes:

- Use `.venv/bin/python` for application imports and smoke runs when possible.
- The project virtualenv may not contain pytest, while the cluster Python may not contain project dependencies such as `tiktoken`. Probe the intended interpreter first. Do not install or mix site-packages without explicit need and approval.
- For syntax-only shell validation, use `bash -n script.sh`; do not submit or execute the SLURM job.
- Network access remains separately restricted even under normal repository work. If an essential download fails for network/sandbox reasons, use the execution tool's approval flow rather than implementing a workaround.

## How `main.py` Fits Together

The file is intentionally monolithic and tightly coupled. Changes to a LUT or attention path normally affect several surfaces:

- Argument normalization and validation: `configure_new_mha`, `configure_lut_comparison_mode`, `configure_weight_sharing`, and `configure_cpu_parallelism`.
- Structures and scratch state: `LUT`, `AttentionHead`, `NewAttentionHead`, `Model`, `BatchElement`, and `SparseGradAccumulator`.
- Construction and topology: `build_LUT_custom`, `build_Model`, and `_share_lut`.
- Index caching: `cache_index_batch`, PE/QKPE caches, Python fallbacks, and Numba equivalents.
- Forward/backward: old attention, soft attention, new attention, FFN/output heads, layer norm, and MHA O LUT.
- Batch size 1 versus `--batch-size > 1`: direct updates versus sparse-gradient collection and `apply_averaged_gradients` replay.
- State/reporting: `save_model`, `load_model`, model stats, hyperparameter printing, W&B config, LUT stats, and analysis scripts.

Never update only the fast Numba path. Keep the NumPy/Python fallback semantically identical; `--fp16` and environments without Numba may use it.

### LUT representation

Each `LUT` holds `S`, `a_arr`, `b_arr`, `bitmasks`, `trees`, and `comparisons_per_bit`.

- `S` is `(physical_trees, 2**total_n_c, y_dim)`.
- `a_arr,b_arr` are flattened raw pairs in bit-major order: `(logical_trees, n_c * comparisons_per_bit)`.
- Each learned bit is the XOR parity of its consecutive raw pairs; backward chooses the raw pair with the smallest absolute margin, then flips its aggregate bit.
- `trees` maps logical comparator slots to physical `S` trees; this is how Final sharing works.
- `n_c` is the number of learned/data-dependent row bits. Tables that add positional bits use a larger `total_n_c`; Final uses `n_c + positional_dim`. Memory is exponential in this total.

### Attention modes

- Default hard attention: historical `AttentionHead.V`, indexed with input-comparison and positional bits.
- `--soft-attention`: decomposed `Score` and `Value` LUTs with softmax weights. This is distinct from new attention.
- `--new-attention`: each head has `Q`, `K`, `V`, and `Final`. Q/K produce `attn_qk_dim` vectors; Final indexes both the V side and Q+K+PE side.
- `--new-MHA`: implies new attention. Each head emits `embedding_dim / num_heads`; outputs are concatenated and a layer-level `MHA_O` LUT maps them back to `embedding_dim`.

New-attention causal pairs only use `j < i`; position zero has no attention contribution.

### Batch and CPU behavior

- Batch size 1 mutates `S` directly in backward.
- Batch size greater than 1 uses `BatchElement` scratch models sharing weights, accumulates sparse gradients, and replays their average.
- `--cpu-cores` is the budget. Zero-valued worker flags derive values from it.
- `--num-workers` parallelizes batch elements; `--replay-workers` parallelizes sparse replay; `--numba-threads` is per worker/kernel.
- `--head-workers` only parallelizes new-MHA heads for batch size 1. Values above 1 require `--new-MHA` and are rejected with batched training.
- Cluster scripts also set BLAS/OpenMP/Numba thread variables. Avoid oversubscription.

## Flag and Configuration Reference

The parser in `main.py` is authoritative. These are the interactions agents most often need.

### Data and task

- `training_data`, `--validation-data`: training and validation input.
- `--tokenizer word`, `--vocab-file`: standard setup for current POS experiments.
- `--POS-task` / `--pos-task`: predicts 18 universal POS labels. Requires word tokenization and rejects `--factored-output`.
- `--pos-cache-dir`: cache for spaCy-derived aligned labels. Current WikiText runs use `nanoGPT_POS/.pos_cache`.
- `<unk>` word IDs still receive POS targets derived from original text; do not break tokenizer/spaCy alignment.

### Shape and LUT capacity

- `--context-size`, `--embedding-dim`, `--num-layers`, `--num-heads`: main model shape.
- `--n-t`, `--n-c`: logical trees and learned row bits for normal LUTs.
- `--attn_qk_dim`, `--attn_qk_n_c`, `--attn_qk_n_t`: new-attention Q/K width, bits, and tree count.
- `--fp16`: float16 `S` tables; saves memory but may select slower fallbacks and changes numerics.
- `--factored-output`: base-256 output split for large LM vocabularies; incompatible with POS.

### Comparison construction

- `--lut-comparison-mode random`: independently sampled distinct `a,b` pairs; default and common in recent full runs.
- `no-overlap`: deterministic disjoint coordinate groups. It overrides `n_t` and `attn_qk_n_t` with the number of possible groups.
- `overlap`: starts with disjoint groups and adds deterministic mixed groups to the requested tree count.
- `--comparisons-per-bit K`: XOR aggregation for learned LUT bits. A learned bit is the parity of `K` random comparisons. Thus `a_arr/b_arr` have `n_c*K` raw comparisons while `S` retains `2**n_c` learned bins. `K>=1`; `K>1` requires random mode. Backward selects the closest raw comparison and flips its logical bit using `r_min // K`. `K=1` is exactly legacy-compatible for seeded pair generation, indexing, input gradients, and `S` updates.

The QKPE side has separate direct/fixed logic and is not multiplied by `--comparisons-per-bit`; aggregation applies to learned LUT comparison construction, including Final's V side.

### New-attention and PE ablations

- `--new-attention`: enables Q/K/V/Final attention.
- `--PE-only`: requires new attention; Q/K contributions are zeroed for a PE+V+Final ablation.
- `--freeze-PE`: skips positional-encoding gradient updates after initialization/load.
- `--PE-init-scale S`: nonnegative scale used for frozen new-attention PE magnitudes.
- `--qkpe-direct-sign`: requires new attention and `positional_dim <= attn_qk_dim`; bit `c` uses `qkpe[c] > 0` instead of an `a,b` comparison.
- `--normalize-qk`: requires new attention; normalizes Q+K per pair before adding raw PE. Target std is `PE_init_scale/sqrt(3)` for frozen PE, otherwise `1/sqrt(3)`.
- `--qk-norm-eps`: positive variance epsilon. Normalization needs at least two relevant QK dimensions.

### New MHA and Final sharing

- `--new-MHA` / `--new-mha`: implies new attention, rejects soft attention, and requires `embedding_dim % num_heads == 0`.
- Head/V/Final output width is `mha_head_dim = embedding_dim // num_heads`; commit `b05d064` fixed these value dimensions. Preserve them in scratch state, kernels, save/load, and stats.
- `--weight-sharing N`: new attention only. Shares physical `Final.S` trees, not Q/K/V, FFN, output, or MHA O. It requires `1 <= N <= n_t` and divisibility. There are still `n_t` logical comparator slots; `trees` repeats `N` physical IDs, making `final_n_t=N` and `final_share_m=n_t/N`.
- Checkpoint shapes depend on architecture, dimensions, sharing, bits, and comparison aggregation. Load with the same structural command.

### Optimization and regularization

- `--warmup-steps`: positive linear warmup length.
- `--max-learning-rate`: positive peak LR. Default is the legacy-compatible `1/sqrt(4001)`.
- `--lr-scheduler inverse-sqrt`: `max_lr * (warmup_steps/step)**inverse_sqrt_alpha` after warmup.
- `--inverse-sqrt-alpha`: nonnegative, default 0.5; one experiment used 0.32.
- `--lr-scheduler cosine`: cosine decay across remaining `max_steps`.
- `--cosine-min-lr-ratio`: final/max LR ratio in `[0,1]`, default 0.01.
- `--layernorm`: pre-norm before attention and FFN.
- `--dropout`: residual-delta dropout during training only.
- `--seed`: seeds Python and NumPy before model/data construction; always set it for comparisons.

### Validation, checkpoints, statistics, and W&B

- `--testing-length`: validation snippets per evaluation.
- `--validation-interval`: also controls periodic model saves and base-stat refreshes.
- `--loss-file`: overwritten at startup; give every run a unique path.
- `--save-model`: NumPy adds `.npz` if omitted; saves periodically and at the end.
- `--load-model`: weights only. It does not resume step, LR schedule, RNG, or loss logging.
- `--save-multiple-checkpoints`: saves step 0 and step-tagged validations. Its introducing commit says “not sure it works”; verify outputs before relying on resume/analysis semantics.
- `--lut-update-stats-file X.npz`: also emits `X_summary.csv`, `X_histogram.csv`, coordinate NPZ/summary, and `j` NPZ/summary/top CSVs.
- `--lut-update-stats-snapshot-steps 1000,10000,...`: positive 1-based cumulative snapshots; steps beyond `max_steps` never fire.
- `--wandb` plus project/entity/name/group/tags/mode/log interval: logs serializable args, train/validation metrics, and LR. Keep names aligned with the actual command.

## Experiment Ledger

This describes what commits and saved `config.txt` files show was attempted. It does not infer conclusions from run names alone; use loss files, stdout, W&B, and generated analyses for outcomes.

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

### Epoch budgets, LR scaling, and the cooldown (torch port, Sep 2026)

- The torch port's training schedule was retuned from scratch: budget in epochs rather
  than steps, peak LR scaled linearly with batch, warmup `6400/batch`, and a terminal
  cooldown. `lut_torch/AGENTS.md` ("Training recipe") has the settings; **`DECISIONS.md`
  at the repo root has the measurements, the wrong turns, and what is still open.**
- Outcome: `new-MHA=on` beats `new-MHA=off` by 0.010-0.017 with 3-7x fewer parameters,
  and reaches 0.9454 in ~57 min where the old schedule reached 0.9445 in 3.30 h.
  Differences *within* `new-MHA=on` are smaller than seed noise and are not a ranking.
- Result trees: `POS_torch_gpu_sweep/` (first sweep, fixed 200k steps),
  `POS_torch_lr_probe*/` and `POS_torch_exp{A,A2,B,C,D,E}/` (the tuning probes),
  `POS_torch_gpu_sweep_v2_E10/` (the rerun). Each run's `config.txt` names its swept axis
  under `swept_*` keys; trust those over directory names, and check for
  `DUE TO PREEMPTION` in the sbatch output before reading a short run as a crash.

## Analysis and Inference Tools

- `plot_lut_update_histograms.py STATS`: plots row counts and discovers matching `j`/coordinate stats and snapshots.
- `analyze_lut_hot_rows.py STATS`: reports frequently updated rows and decoded row structure.
- `compare_lut_table_stats.py RUN_A RUN_B`: compares coverage summaries.
- `compare_lut_checkpoint_rows.py CKPT...`: compares changed rows across first-last, first-all, or consecutive checkpoints.
- `analyze_lut_model_tables.py MODEL.npz`: learned-table distributions and similarity.
- `analyze_multiseed_metrics.py RUN_DIR`: aggregates seed loss CSVs.
- `compare_run_validation.py RUN_DIR...`: scores any mix of `main.py` `.npz` and `lut_torch` `.pt` checkpoints on one pinned validation window set, reading each run's structural flags from its `config.txt`. Use this before believing any accuracy or loss difference read off two runs' `loss.csv` — see "Comparing runs" below.
- `infer_lut_pos.py RUN_DIR --once 'text'` or `--infer-ambiguate`: LUT POS inference.
- `infer_gpt_pos.py CHECKPOINT_DIR`: baseline inference.
- `pos_majority_baseline.py`: empirical majority-POS baseline.
- `comp_txt_pos.py`: binary-token/POS-cache alignment check.

Compatibility warning: `infer_lut_pos.py` has its own reduced parser and currently lacks newer structural flags including `--new-MHA`, `--weight-sharing`, frozen/normalized PE, and `--comparisons-per-bit`. It cannot correctly reconstruct checkpoints using those flags until updated. Add every new structural flag to inference reconstruction.

## Running Experiments Safely

```bash
cd /cs/labs/roys/yuval.rubin/LUT/LUT
source .venv/bin/activate
```

Run scripts should create timestamped directories, redirect temp/W&B state there, save the exact shell-escaped command in `config.txt`, and link SLURM output. Preserve old configs as provenance; copy a script for a new experiment.

Before a long run:

1. Align directory, W&B name/tags, resource count, and actual flags.
2. Run parser/tests and a tiny smoke first.
3. Use unique loss/model/stats paths.
4. Check printed normalized hyperparameters; no-overlap rewrites counts and new MHA derives dimensions/workers.
5. Confirm the first validation/checkpoint/stat output before scaling.

Minimal POS/new-attention smoke:

```bash
python main.py nanoGPT_POS/data/wiki2/train_original_10_freq.txt \
  --validation-data nanoGPT_POS/data/wiki2/val_original_10_freq.txt \
  --pos-cache-dir nanoGPT_POS/.pos_cache \
  --vocab-file nanoGPT_POS/data/wiki2/vocab_file_10_freq.txt \
  --tokenizer word --POS-task --new-attention \
  --context-size 8 --embedding-dim 16 --num-layers 1 --num-heads 1 \
  --n-t 2 --n-c 3 --attn_qk_dim 4 --attn_qk_n_c 3 --attn_qk_n_t 2 --comparisons-per-bit 1 \
  --testing-length 2 --validation-interval 2 --max-steps 3 \
  --warmup-steps 2 --loss-file /tmp/lut_pos_smoke.csv \
  --seed 42 --cpu-cores 1
```

This needs local data/vocab, dependencies, and POS cache or spaCy model. First Numba compilation may dominate a tiny run.

## Comparing Runs

`loss.csv` from two runs is not directly comparable, and the difference this hides is
larger than most of the differences experiments are launched to measure.

Each run fixes its validation windows once at startup by sampling `--testing-length`
random starts. At the historical setting of 100 windows that is only ~2,900 scored
tokens, whose 95% sampling interval on accuracy is **+/-1.1 points**. Two window sets
drawn independently differ enough in difficulty that the *same* checkpoint scored 0.9201
on one and 0.9374 on the other — a 1.7-point spread with no model difference at all.
This is what made the first `lut_torch` GPU run
(`results/POS_torch_gpu/20260906-142646_torch_gpu_WS8_bs1_sgd`) look better than the
`main.py` run it was ported from (`results/POS_new_attn/20260814-101642`); scored on the
same 1,000 windows the two land at identical accuracy (0.928331 each, 29,134 tokens),
agreeing on 95.7% of individual predictions with 579 tokens right each way.

So:

- Compare checkpoints with `compare_run_validation.py`, never two `loss.csv` tails.
- Runs launched from `run_files/` now use `--testing-length 1000` (+/-0.35 points).
  The torch scripts pair it with `--validation-batch-size 100`, which keeps the 10x
  window count roughly free; `main.py` scores one window at a time and pays ~23s per
  validation, so check the interval before adding validations to a long run.
- Both trainers now draw the validation windows the same way (Python `random`), so at
  the same seed they score the same snippets and build the same model. See invariant 1 in
  `lut_torch/AGENTS.md`. Training snippet order and dropout masks still diverge, so the
  runs themselves are not step-for-step identical.

## Validation Expectations

Use the smallest relevant checks first:

```bash
python -m py_compile main.py
python test_vectorize.py
python test_lut_update_stats.py
python test_new_mha.py
python test_compare_lut_checkpoint_rows.py
python -m pytest -q test_xor_comparisons.py
```

For the torch port:

```bash
.venv/bin/python test_torch_lut_primitive.py
.venv/bin/python test_torch_parity.py
.venv/bin/python test_torch_training.py
```

The venv has torch but no pytest, while the cluster python has pytest but no torch, so these
three import `parametrize`/`run_all` from `lut_torch/_testing.py` and run under either
(`python FILE` uses the shim runner; `python -m pytest FILE` uses real pytest). They import
`main.py` to compare against it, so they need the venv.

Several are executable assertion scripts rather than pytest tests; `test_xor_comparisons.py` requires pytest. `test_prange.py` is a performance probe and can be expensive. Numerical/parallel changes should compare Numba/fallback, batch-1/replay, forward/backward/`S` updates, save/load, model stats, and seeded single-/multi-thread behavior.

## Rules for Future Changes

- Preserve uncommitted work and research artifacts. Never clean results, checkpoints, caches, or W&B directories unless explicitly asked.
- Do not edit historical run configs; make a new script/config.
- Structural flags must reach parser validation, printed/W&B config, run config, inference reconstruction, save/load compatibility, model stats, batch scratch state, and tests.
- New LUT names must reach update stats and analysis-tool discovery.
- Row-index changes require auditing every `jbar` flip, `r_min`, Python fallback, Numba kernel, sparse replay path, and analysis decoder.
- Shared-tree updates must avoid races between logical slots mapped to one physical tree.
- A checkpoint does not store a complete executable configuration; load it with matching structural flags.
- Keep POS token/label alignment intact; never retokenize independently and guess positions.
- The README CLI table predates many flags. Prefer `main.py --help`, this file, and saved configs.
