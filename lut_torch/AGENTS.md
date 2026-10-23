# AGENTS.md — `lut_torch/`

Read this before changing anything in this directory. The repository-wide file at
`../AGENTS.md` still governs cluster/sandbox rules, the data and POS cache, `main.py`, and
the analysis tooling; this file governs the torch codebase. Runtime and memory
measurements live in `PERFORMANCE.md` next to this file; the measurements behind the
training recipes live in `../DECISIONS.md`.

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
| `modules.py` | `LUTTable`, `NewAttentionHead`, `StackedHeads`, `LutLayer`, `LutModel`. Construction order mirrors `main.build_Model` exactly. |
| `comparisons.py` | a/b comparison-pair construction and PE init, RNG-order-faithful to `main.py`. |
| `config.py` | `LutConfig`: structural flags, validation, derived fields. |
| `data.py` | Tokenizer/POS loading (shares `main.py`'s cache filenames), `.bin` token files, window batching. |
| `train.py` | CLI trainer: loop, validation, optimizers, schedule, checkpoints, W&B. |
| `kernels.py` | Fused-Numba CPU backend for `LUTFunction` (see "Kernels"). |
| `stacking.py` | Converts per-head checkpoints to the `--stack-heads` layout. |
| `graphs.py` | CUDA-graph capture/replay of forward+backward. |
| `bench_cpu.py` | CPU scaling benchmark (`python -m lut_torch.bench_cpu`). |
| `run_index.py` | Finds trained runs from `results/*/*/config.txt`. Stdlib only — no torch. |
| `inference.py` | Loads a checkpoint (LUT or nanoGPT baseline) and samples from it. |
| `chat_server.py` + `chat_ui/` | Web chat interface (`python -m lut_torch.chat_server`). |
| `_testing.py` | pytest shim so tests run on either interpreter here. |

Tests live at the repository root; see "Testing".

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
   `main.load_training_data`'s `testing_input_data` draw for draw, so at the same seed both
   trainers hand `build_Model`/`LutModel` the same Python *and* NumPy streams and construct
   the same model. Switching it back to `np.random.randint` (as it was before) offsets the
   NumPy stream by `--testing-length` draws and every table in the model differs — with no
   error and no test failure, because the parity suite never goes through `train.py`.
   Anything new that draws before the model is constructed has to be mirrored in `main.py`
   or it breaks this the same silent way.

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

6. **`SnippetDataset.tokens` keeps its input dtype**: int32 from text, or a uint16 memmap
   from `.bin`. Only the gathered window is widened to int64, in `batch()`, so callers get
   int64 exactly as before. Do not convert the whole stream with `np.asarray(...,
   np.int64)`: for WT103 that is a ~1 GB copy per process, and it would load the whole
   memmap into memory.

## Table gradients

A LUT's gradient touches one row per **visit** — one per (batch, position-or-pair, tree).
Sparse only pays off while visits are fewer than the table's rows; the Final table is
indexed per causal pair per tree, so at context 32 a single batch element already exceeds
its row count. `--table-grad auto` (the default) picks dense once visits reach the row
count, which is right in both directions. Force `sparse` only when a dense gradient will
not fit. Getting this wrong looks like bad scaling, not a bug — unconditional sparse once
cost 19x at batch 32 (`PERFORMANCE.md`).

## Performance in brief

Full measurements and history: `PERFORMANCE.md`.

- **GPU:** use `--stack-heads` and `--cuda-graphs` together (8.3x over neither at batch 1).
  Both attack kernel-launch count, which was the binding constraint. No custom CUDA
  kernels; `--kernels` is a no-op on GPU. GPU runs are not bit-reproducible
  (`index_add_` atomics); the parity suite stays CPU-only.
- **CPU:** one knob, `--cpu-cores` (ATen's intra-op pool); `main.py`'s four worker flags
  are gone. Batch size, not thread count, makes cores usable. Set `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` to match. Several narrow processes beat one wide one (4x4 beat 1x16
  by 1.38x; 8x2 collapses).
- **vs `main.py`:** slower at batch 1 (0.66x), 2.4x faster per sequence at batch 32. Do not
  compare at batch 1 and conclude the port is slow.
- If you touch the primitive, re-measure across batch sizes; several past regressions only
  showed at one batch size.

## Flags

`train.py --help` is authoritative. Notes on the ones with sharp edges:

- `--batch-size` — the CPU-utilisation knob, and it also changes the gradient (it averages
  over the batch).
- `--optimizer {sgd,adam,adamw}` — `sgd` is the `main.py`-equivalent default.
  `--momentum` and `--weight-decay` affect dense parameters only.
- `--gradient-pair-selection {closest,cosine-distance}` — `cosine-distance` is meaningfully
  more expensive: its choice depends on the upstream gradient and on the table rows, so it
  varies per pair and per tree and must be assembled at expanded shape.
- `--new-MHA` is optional. Without it, each head emits the full `embedding_dim` and head
  outputs are summed into the residual with no `MHA_O` table, exactly as in `main.py`.
- `--load-model` resumes properly (model + optimizer + scheduler + step + RNG).
  `--reset-optimizer` loads weights only.
- `--output-code-dim D` / `--output-code {learned,binary,random}` — the output table emits
  `D` numbers instead of one score per class, and a `(V, D)` code matrix `C` maps them to
  logits, `logits = y @ C.T`, so the table costs `n_t * 2**n_c * D` entries rather than
  `n_t * 2**n_c * V`. **This is not a relaxation:** scoring class `k` as independent
  per-coordinate bit comparisons, `sum_b [c_kb log sigma(y_b) + (1-c_kb) log sigma(-y_b)]`,
  differs from `sum_b c_kb y_b` only by a `k`-independent term the softmax cancels
  (`test_torch_output_code.py`). It caps the logit matrix at rank `D`, which argmax may not
  feel but perplexity does — report validation loss, not just accuracy.
  **Use `learned`, `D = 112`, with `--output-code-bias`** for the LM. `binary` and `random`
  are frozen-code controls, not options; freezing is what costs, not Hamming distance.
  (`ecoc_masks`' greedy construction leaves `binary`'s minimum distance at 2; fixing it is
  not expected to matter.) Numbers: DECISIONS.md L1, L3.
- `--output-code-bias` — adds a per-class bias, `logits = y @ C.T + b`. Without it class
  priors must lie in `col(C)`, which a frozen code cannot arrange. Costs `V` parameters.
  Requires `--output-code-dim`.
- `--output-code-lr-scale` — multiplies the learning rate of `C` and `b` only (a step moves
  the table's contribution to the output by roughly `n_t x` nominal, `C`/`b` by `1x`). No
  `main.py` counterpart, so it cannot enter `test_torch_parity.CONFIGS`; parity is protected
  because the code matrix is drawn last and only when the flag is set, leaving every
  existing configuration bit-identical (`test_default_head_unchanged`).
- `--lm-validate-last-position-only` — LM validation scores every position by default;
  this restores `main.py`'s last-position-only scoring. **The two modes are not
  comparable** (the last position has the only full context, so its loss is lower). LM
  numbers recorded before the change were measured the old way.
- `--best-checkpoint-metric {auto,acc,loss}` — `auto` selects on accuracy for POS and on
  validation loss for LM.
- `--save-model` also writes `..._best.pt` at the best validation metric so far; the run
  summary carries `val/best_acc` / `val/best_step` / `val/best_epoch`. The final checkpoint
  is not necessarily the best one.
- **`<unk>` in LM validation.** The LM trains on `<unk>` as an ordinary target (POS masks
  targets whose *input* was `<unk>`; LM does not). `evaluate` returns both
  `loss`/`acc` over every target and `loss_no_unk`/`acc_no_unk` with `<unk>` removed; both
  reach `loss.csv` and W&B. Use the no-unk pair to compare word-tokenizer runs.
  **Word tokenizer only:** the triple is NaN under BPE, where id 0 is a real token (`"!"`
  in GPT-2). `load_datasets` records this as `dataset.unk_id = None`.
- **Pre-tokenized `.bin` input** (DECISIONS.md W3). A positional training path or
  `--validation-data` ending in `.bin` is memory-mapped instead of read and encoded. It
  needs a `<file>.bin.json` manifest (`format: lut-token-bin-v1`, `tokenizer`, `dtype`,
  `n_tokens`, ...; written by `prepare_wikitext103.py`), checked by
  `read_token_bin_manifest`: the file size must match and `tokenizer` must equal
  `--tokenizer`. Both splits must be `.bin` or both text, and POS rejects `.bin` because it
  needs the text.
- **Per-word loss for BPE** (DECISIONS.md W2). `evaluate` returns `loss_per_word` = summed
  NLL of every scored target / number of scored targets that start a whitespace word, plus
  `words`. `data.WordTables` decides word starts from each token's bytes: it has a
  non-whitespace byte AND (it starts with whitespace OR the previous token ends with
  whitespace). Over a whole stream the count equals `len(text.split())` exactly
  (`test_token_bin.py`). `loss.csv` gets `loss_per_word, perplexity_per_word` as its last
  two columns (NaN for the word tokenizer). For BPE runs, `perplexity_per_word` is the
  number to compare.
- `--tree-weights` — every table gets a learned weight per tree slot,
  `y = sum_t w[t] * S[trees[t], j_t]`. Initialised to 1, so a fresh model is the unweighted
  one. One weight per *logical* slot, so slots sharing a physical tree under
  `--weight-sharing` weigh separately. Trained in their own dense-optimizer group with
  momentum but **no weight decay** (decay pulls towards 0, not 1). A weighted call never
  reaches the Numba kernels. Off by default; checkpoints without `tree_weights` keys will
  not load into a model built with it.
- `--lut-comparison-mode` accepts only `random`; it exists so `main.py` commands port over.
- `--device {cpu,cuda,cuda:N,auto}` — `--kernels` is silently a no-op on GPU.
- `--stack-heads {auto,on,off}` — run a layer's heads as 4 `lut_lookup` calls instead of
  `4 * H` (`StackedHeads`). `auto` is on whenever `num_heads > 1`. **It changes the
  parameter layout**, so state dict keys differ; `--load-model` converts a per-head
  checkpoint automatically (`stacking.py`) and discards optimizer state. The forward is
  bit-identical to the unstacked path and single-layer gradients are too; deeper gradients
  differ by 1-3 float32 ULP because fusing `H` gathers reassociates the scatter-add. That
  is why the unstacked path still exists and why `test_torch_parity.py` builds unstacked.
- `--cuda-graphs {auto,on,off}` — replay forward+backward from a single launch
  (`graphs.py`). `auto` is on for CUDA, a no-op on CPU. Numerically identical. Forces
  `--table-grad dense`; refuses an explicit `--table-grad sparse`. The optimizer and LR
  scheduler stay eager on purpose — see the module docstring.

**Schedule flags.** Why each exists is in DECISIONS.md Part I; mechanics only here.

- `--epochs` — `max_steps = ceil(epochs * positions / (batch * CS))` in `derive_schedule`,
  which also re-paces warmup, validation cadence and logging. Never budget in steps
  (DECISIONS D1).
- `--lr-batch-scaling {none,linear,sqrt}` / `--lr-reference-batch-size` (default 1) — the
  batch size `--max-learning-rate` is quoted at. Use `linear` (DECISIONS D3).
- `--lr-decay-start` (t0) — inverse-sqrt decay begins here; LR holds at the peak from end
  of warmup until t0. It decouples ramp length from decay height (in `(W/s)**alpha` the
  warmup length otherwise sets both). Defaults to the warmup length, bit-identical to the
  historical curve. Must be ≥ the resolved warmup (`derive_schedule` refuses otherwise).
  Depth before cooldown is reported as `lr_decay_depth`.
- `--lr-cooldown-frac` / `--lr-cooldown-min-ratio` — linear ramp to the min ratio over the
  tail of the run, on top of any scheduler. At 0 the schedule is bit-identical to
  `main.get_learning_rate` (`test_schedule_matches_main`).
- `--warmup-frac` / `--warmup-min-steps` — warmup as a fraction of `max_steps`, which holds
  warmup tokens and end depth fixed across batch sizes and budgets.
  `--warmup-min-steps` deliberately breaks the depth match on short runs.
- `--validation-interval-epochs` / `--wandb-log-points` — cadence measured against data, so
  curves from different batch sizes land on the same epoch values and get the same W&B
  smoothing.

**Pass each flag once in run scripts.** argparse keeps the last value silently; the LM
Part II scripts inherited `--lr-cooldown-frac 0.2` from the POS template this way.

**Not carried over** (still available in `main.py`): old hard attention, `--soft-attention`,
`--factored-output`, `--fp16`, `no-overlap`/`overlap` comparison modes,
`--halve-lr-at-20k-40k`, `--save-multiple-checkpoints`, and the `LUTUpdateStats`
instrumentation.

## Training recipes

Change these only against `../DECISIONS.md`, which has every number behind them.

**POS** (DECISIONS.md Part I; batch 32, `heads 4 / new-MHA / weight-sharing 8`):

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

Batch size is a speed knob, not an accuracy knob. Prefer `heads 4` over 8 (1.71x faster,
no measurable gain). Keep a terminal cooldown but do not tune its shape. Noise floors:
**0.0005** accuracy / 0.0030 val loss between jobs at identical config and seed,
**0.0012-0.0031** between seeds (`--seed` also draws the validation windows). Val loss is
the more sensitive metric; train loss says nothing useful about budget.

**LM, WikiText-2 word-level** (DECISIONS.md Part II; batch 32, `n_c 8`, ctx 32, emb 64,
h4, ws8, new-MHA):

```
--output-code learned --output-code-bias --output-code-dim 112
--num-layers 32                     # or 48; depth is where the freed parameters go
--epochs 30                         # not more at depth -- it feeds overfitting
--lr-scheduler cosine --cosine-min-lr-ratio 0.01
--lr-cooldown-frac 0.2
--warmup-frac 0.0095
--max-learning-rate 0.00125         # x batch 32 = 0.04 effective
--lr-batch-scaling linear --lr-reference-batch-size 1
--validation-interval-epochs 0.1
```

Unlike POS, the LM is budget-starved and the anneal shape matters (cosine beats
inverse-sqrt). Re-bracket peak LR whenever the budget changes. WikiText-103 / GPT-2 BPE
settings are in DECISIONS.md Part III.

## Talking to a trained model

```bash
srun --gres=gg:g0:1 --mem=16g -c4 --time=2:00:00 --pty run_files/chat_server.sh
python -m lut_torch.chat_server --device cpu    # or wherever the venv works
```

A browser chat over LM checkpoints: stdlib `http.server` plus static files under
`chat_ui/`, nothing to install. Use `srun`, not `sbatch` — it is a service you sit in
front of.

Three facts about the model decide the design, and all three are easy to get wrong:

1. **It is a next-token model, not an assistant.** A "conversation" is one growing piece
   of text: the UI sends the transcript verbatim as the prompt, and the reply is the
   continuation.
2. **The context is fixed and there is no KV cache.** A prompt shorter than
   `context_size` goes at positions `0..n-1` and the prediction is read at `n-1`; longer,
   and the window slides. Trailing padding is invisible because attention is causal
   (`test_short_prompt_ignores_padding`). Left-padding would be wrong twice: the PE is
   absolute, and position 0 receives no attention output at all.
3. **`model.eval()` is mandatory.** `LutLayer._residual_dropout` is live in `train()`
   mode, so a forgotten `eval()` samples from a perturbed model — and nothing raises.

Listing runs never opens a checkpoint: `run_index.py` reads `config.txt` and `loss.csv`
only, and classifies a run by **the recorded command**, not the directory name.

**Both engines.** The chat also serves the `gpt_baseline_train.py` transformer (`engine`
`nanogpt`); legitimate because both share corpus, tokenizer, context and — at the same
seed — validation windows. `gpt_baseline_train.save_checkpoint` writes
`kind: 'nanogpt_lm'`; its absence means LUT. Generate through
`gpt_baseline_train.FullLogits`, never `GPT.forward` directly (which keeps only the last
position). The GPT state dict loads by copy, not `assign=True` — see the comment in
`inference.py`. A live run rewrites `model_best.pt` on every improvement, so
`load_checkpoint_file` retries and the server falls back to the periodic copy. `<unk>` is
suppressed during sampling by default (word tokenizer only).

**Two tokenizers.** Every word-vocabulary fact is false under GPT-2 BPE, and *quietly*:

| word vocabulary | GPT-2 BPE |
|---|---|
| id 0 is `<unk>` | id 0 is `!` — suppressing it bans an ordinary token |
| id 1 is `<eol>` | id 1 is `"` — the trailing-`<eol>` trim **deletes a real character** from any prompt ending in a quote |
| `tokens_from_ids` | does not exist; tiktoken raises `AttributeError` |
| a piece needs a leading space added | pieces already carry it (`['The', ' first', ' game']`) |
| `enc.encode` takes anything | raises on `<|endoftext|>`, so arbitrary chat input is a 500 unless `disallowed_special=()` |

So all of those facts live behind `make_tokenizer_adapter()` in `inference.py`, and
nothing outside the word adapter may act on id 0 or id 1. The BPE `stream()` is a
*byte*-level incremental decoder, so a character split across two tokens does not become
two replacement characters.

**Headline metric depends on the tokenizer.** GPT-2 emits 1.1664 tokens per whitespace word
on this corpus, so a BPE run's per-token perplexity is ~2.5x rosier than its per-word one.
`RunInfo.headline_metric` picks `per_word` for BPE and `per_token` for word runs, and
`headline_fallback` flags a BPE run that has not written a per-word value yet.

## Testing

```bash
.venv/bin/python test_torch_lut_primitive.py
.venv/bin/python test_torch_parity.py
.venv/bin/python test_torch_training.py
.venv/bin/python test_torch_kernels.py
.venv/bin/python test_torch_stacking.py
.venv/bin/python test_torch_graphs.py       # skips itself without a GPU
.venv/bin/python test_torch_output_code.py
.venv/bin/python test_torch_chat.py         # its torch-free half runs anywhere
.venv/bin/python test_token_bin.py          # .bin input, word-start counting, per-word loss
.venv/bin/python test_gpt_baseline.py       # GPT baseline shares windows/evaluate with lut_torch
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
| `test_torch_graphs` | CUDA-graph replay vs eager over *consecutive* steps, dropout redraw, and the zero_grad misuse guard |
| `test_torch_output_code` | coded output head: the linear-readout equivalence, code construction, default head unchanged |
| `test_torch_chat` | run discovery, LM/POS classification, engine detection, incremental decoding, sampling filters, the causal-padding assumption, checkpoint-kind dispatch, server routes |
| `test_token_bin` | `.bin` manifests and memmap loading, word-start counting, per-word loss |
| `test_gpt_baseline` | the GPT baseline scores the same windows as a `lut_torch` run at the same seed |

`test_torch_stacking` is why stacking is safe to trust: parity pins unstacked against
`main.py`, and stacking pins stacked against unstacked.

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
chosen so every iteration owns its output cells outright — see each kernel's docstring. The
paired-group backward (`_k_grad_paired`) parallelises over *source position*, not pair, using
a CSR inverse of `pair_index` (`NewAttentionHead.final_pair_ptr`/`final_pair_list`, built once
at construction) so that several pairs accumulating into one source position never race.

Every kernel has a serial mirror (`_k_*_serial`), generated at import time from the parallel
kernel's own source (`prange` → `range`, `parallel=True` dropped), so there is exactly one
body per kernel to keep correct. `_pick` chooses serial below `_SERIAL_BELOW_ITEMS` (4096 work
items) and parallel, thread-capped by `_threads_for`, above it.

**`fastmath` is never used.** It reorders float reductions, which risks the bit-exact parity
`test_torch_parity` exists to guarantee, and it measured no net win.

The kernels are 1.8-3.7x faster per call in isolation but showed no consistent end-to-end win
on a shared, noisy node; re-measure on an exclusive allocation before trusting either
direction (`PERFORMANCE.md`).

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
- **POS inference tools** (`infer_lut_pos.py`, the `analyze_*` / `compare_*` scripts) read
  `main.py`'s `.npz` files and cannot load these `.pt` checkpoints, except
  `compare_run_validation.py`, which reads both. LM sampling is in `inference.py`.
- **Old attention** (hard and soft) is not ported. If it is ever wanted, it is the same
  primitive with three margin groups (Q, K, PE) instead of Final's two — `LutGroup` already
  covers the shape.
