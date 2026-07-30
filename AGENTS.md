# AGENTS.md

Read this before changing the repo. Keep future edits small and validate with a smoke run when possible.

## What This Repo Is

This is a NumPy/Numba research codebase for a lookup-table language model, based on the SNN/LUT transformer idea. The active implementation is `main.py`; it trains either next-token prediction or a part-of-speech prediction task. The repo also contains a fork/copy of `nanoGPT_POS/` used as a transformer POS baseline.

Most current experiments are POS tagging on WikiText-style data with a word tokenizer. POS mode uses word IDs as inputs and predicts 18 universal POS labels as outputs.

## Important Files

- `main.py`: main model, training loop, CLI, save/load, validation, W&B logging, LUT update stats.
- `word_tokenizer.py`: word vocabulary/tokenization and POS label helpers.
- `run_files/run_POS_one.sh`: current single-run SLURM template for large new-attention POS experiments.
- `run_files/run_POS_multiseed_new_attn.sh`: launches seeds 42-45 across SLURM tasks for new-attention POS.
- `run_files/run_GPT_POS.sh`: nanoGPT/POS baseline script name is historical; inspect before trusting.
- `analyze_lut_model_tables.py`, `analyze_lut_hot_rows.py`, `compare_lut_table_stats.py`, `plot_lut_update_histograms.py`: post-hoc analysis for saved LUTs and row-update coverage.
- `results/`: experiment outputs; configs are useful provenance. Do not treat as source code.
- `saved_models/`, `out/`: checkpoints. Some `out/` and `nanoGPT_POS/` files are untracked in Git.

## Main Architecture Decisions

- LUTs are represented by `LUT.S`, `a_arr`, `b_arr`, `bitmasks`, and `trees`. `S` rows are updated directly during backprop/replay.
- There are three attention modes:
  - default old hard attention: one `V` LUT indexed by query/key/position bits.
  - `--soft-attention`: decomposed `Score` and `Value` LUTs with softmax weighting.
  - `--new-attention`: separate `Q`, `K`, `V`, and `Final` LUTs. Q/K use `--attn_qk_dim`, `--attn_qk_n_c`, and `--attn_qk_n_t`; `Final` combines value rows with QK+positional bits.
- POS task intentionally disables `--factored-output`; it uses `StandardOutputHead` with `vocab_size_output = len(POS_LABELS)`.
- Batch size > 1 uses `BatchElement` scratch models and `SparseGradAccumulator`; gradients are replayed as averaged sparse LUT updates in `apply_averaged_gradients`.
- Numba kernels are the fast path for `float32` LUT tables. `--fp16` can save memory but may fall back to slower Python/NumPy paths in places.
- `--layernorm` is pre-norm before attention and FFN. `--dropout` applies to residual deltas during training only.

## Recent Work / Experiment Context

- New attention was added in stages: forward path, backward path, update-stats integration, and model-size accounting fixes. When changing attention, keep all four surfaces aligned: build/save/load, forward, backward, and batched replay.
- LUT update statistics were added to understand whether tables actually receive broad row coverage. `--lut-update-stats-file path.npz` also writes summary and histogram CSVs.
- Recent large POS experiments use `--new-attention`, `--POS-task`, 6 layers, 4 heads, `n_t=64`, `n_c=6`, context 32, dropout 0.2, word vocab from `nanoGPT_POS/data/wiki2/vocab_file_10_freq.txt`, and cached POS data in `nanoGPT_POS/.pos_cache`.
- A recent single-seed run at `results/POS_new_attn/20260629-231508` reached about validation loss `0.398`, perplexity `1.49`, and accuracy `0.869` at step 99,000. This is a useful sanity target, not a guaranteed benchmark.
- Multi-seed new-attention scripts currently vary seeds 42-45 and often use positional/QK dimensions around 5 in the script; check `config.txt` inside each result directory for exact executed flags.

## How To Run Safely

Use the repo virtualenv when available:

```bash
cd /cs/labs/roys/yuval.rubin/LUT/LUT
source .venv/bin/activate
```

Small POS smoke run:

```bash
python main.py nanoGPT_POS/data/wiki2/train_original_10_freq.txt \
  --validation-data nanoGPT_POS/data/wiki2/val_original_10_freq.txt \
  --pos-cache-dir nanoGPT_POS/.pos_cache \
  --vocab-file nanoGPT_POS/data/wiki2/vocab_file_10_freq.txt \
  --tokenizer word --POS-task --new-attention \
  --context-size 8 --embedding-dim 16 --num-layers 1 --num-heads 1 \
  --n-t 4 --n-c 4 --attn_qk_dim 4 --attn_qk_n_c 4 --attn_qk_n_t 4 \
  --testing-length 5 --validation-interval 5 --max-steps 6 \
  --loss-file /tmp/lut_smoke_loss.csv --seed 42 --cpu-cores 1
```

For SLURM runs, prefer editing a copy of an existing `run_files/*.sh` script and make sure result paths, W&B names, and `config.txt` capture the full command.

## Cautions For Future Agents

- `main.py` is long but tightly coupled. Avoid broad refactors unless asked.
- If adding/changing CLI flags, update the hyperparameter printout and W&B config behavior.
- If changing model state, update both `save_model` and `load_model`.
- If changing attention table indexing, also update stats collection and analysis scripts that infer LUT names.
- Preserve existing experiment outputs and checkpoints; they are research artifacts.
- `rg` may not be installed on the cluster; use `grep`/`find` as fallback.
