# LUT Language Model

A lookup-table (LUT) based language model that replaces matrix multiplications with learned discrete tables indexed by binary comparisons. Includes a standard transformer baseline via [nanoGPT](https://github.com/karpathy/nanoGPT) for comparison.

---

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [LUT Model](#lut-model)
  - [Quick Start](#quick-start)
  - [All CLI Options](#all-cli-options)
  - [Saving and Loading Weights](#saving-and-loading-weights)
  - [Reproducibility (Random Seed)](#reproducibility-random-seed)
- [Transformer Baseline (nanoGPT)](#transformer-baseline-nanogpt)
  - [Quick Start](#quick-start-1)
  - [Configuration](#configuration)
- [Comparing the Two Models](#comparing-the-two-models)

---

## Installation

**Python requirements** (LUT model):

```bash
pip install -r requirements.txt
# installs: numpy, numba, tiktoken, tqdm
```

**PyTorch** (nanoGPT only):

```bash
pip install torch
```

Numba is used for JIT-compiled kernels and is highly recommended — the model will fall back to pure NumPy if Numba is unavailable, but will be significantly slower.

---

## Data Preparation

The WikiText-2 dataset is the default benchmark. Text files (`wiki2_train.txt`, `wiki2_val.txt`) and pre-tokenized binary files (`wiki2_train.bin`, `wiki2_val.bin`) are included in the repo.

### Using text files (LUT model)

The LUT model can read `.txt` files directly and tokenizes them on the fly:

```bash
python main.py wiki2_train.txt --validation-data wiki2_val.txt
```

### Pre-tokenizing to binary (optional, faster startup)

```bash
python tokenize_data.py wiki2_train.txt wiki2_train.bin
python tokenize_data.py wiki2_val.txt   wiki2_val.bin
```

The binary format stores `int32` token IDs with a 12-byte header (magic + vocab_size + num_tokens).

### Using your own data

Any UTF-8 text file works. For large corpora, pre-tokenize to `.bin` for faster loading.

```bash
python tokenize_data.py my_corpus.txt my_corpus.bin
python main.py my_corpus.bin --validation-data my_val.bin
```

---

## LUT Model

### Quick Start

```bash
# Minimal run — trains on wiki2, evaluates every 10k steps
python main.py wiki2_train.txt --validation-data wiki2_val.txt
```

Progress is shown with a live tqdm bar displaying training perplexity, validation perplexity, and current learning rate. Validation loss is appended to `loss.csv` at each evaluation.

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `training_data` | *(required)* | Path to training text or binary file |
| `--validation-data` | None | Validation file; if omitted, samples from training data |
| `--tokenizer` | `gpt2` | tiktoken encoding name, or `word` for word-level tokenizer |
| `--vocab-file` | `word_vocab.txt` | Vocab file when using `--tokenizer word` |
| `--context-size` | 32 | Sequence length (tokens) |
| `--embedding-dim` | 32 | Token embedding dimension |
| `--positional-dim` | 4 | Positional encoding dimension |
| `--num-layers` | 6 | Number of transformer-like layers |
| `--num-heads` | 4 | Attention heads per layer |
| `--n-t` | 16 | Number of LUT trees per head/layer |
| `--n-c` | 6 | Number of comparison bits per tree (LUT depth = 2^n_c bins) |
| `--max-steps` | 100000000 | Training steps |
| `--validation-interval` | 10000 | Steps between validation runs |
| `--testing-length` | 10000 | Number of snippets per validation run |
| `--temperature` | 0.4 | Sampling temperature for text generation |
| `--loss-file` | `loss.csv` | Output path for loss/perplexity log |
| `--batch-size` | 1 | Sequences per training step |
| `--factored-output` | off | Factored unembedder (memory-efficient for large vocab) |
| `--fp16` | off | Use float16 for LUT S-tables (halves memory) |
| `--soft-attention` | off | Decomposed Score+Value LUTs with softmax weighting |
| `--layernorm` | off | Pre-norm LayerNorm before attention and FFN |
| `--dropout` | 0.0 | Dropout rate on residual stream (0.0 = disabled) |
| `--seed` | None | Random seed for reproducibility |
| `--save-model` | None | Save weights to this `.npz` path at each validation and end |
| `--load-model` | None | Load weights from this `.npz` path before training |

**Example — larger model with regularization:**

```bash
python main.py wiki2_train.txt --validation-data wiki2_val.txt \
    --embedding-dim 64 --num-layers 6 --num-heads 4 \
    --n-t 16 --n-c 6 --context-size 64 \
    --layernorm --dropout 0.1 \
    --batch-size 4 \
    --validation-interval 5000 \
    --loss-file run1_loss.csv
```

**Example — soft attention (transformer-style scoring):**

```bash
python main.py wiki2_train.txt --validation-data wiki2_val.txt \
    --soft-attention --layernorm
```

**Example — fp16 to reduce memory (useful for large vocabularies):**

```bash
python main.py wiki2_train.txt --validation-data wiki2_val.txt \
    --fp16 --factored-output
```

### Saving and Loading Weights

Save model weights to a `.npz` file automatically during training:

```bash
python main.py wiki2_train.txt --save-model checkpoints/my_model
# Saves to checkpoints/my_model.npz at every --validation-interval and at end of training
```

Resume training from a saved checkpoint:

```bash
python main.py wiki2_train.txt --load-model checkpoints/my_model.npz
```

> **Note:** The checkpoint stores weights only. The model architecture must match the hyperparameters used when saving (same `--embedding-dim`, `--num-layers`, `--num-heads`, `--n-t`, `--n-c`, etc.).

### Reproducibility (Random Seed)

Set `--seed` to make all stochastic components (weight initialization, training data sampling, dropout masks) deterministic:

```bash
# Two runs with the same seed produce identical loss curves
python main.py wiki2_train.txt --seed 42 --max-steps 1000
python main.py wiki2_train.txt --seed 42 --max-steps 1000
```

---

## Transformer Baseline (nanoGPT)

A standard GPT-style decoder transformer is set up in the `nanoGPT/` subdirectory using [nanoGPT](https://github.com/karpathy/nanoGPT). The pre-tokenized wiki2 data is already linked at `nanoGPT/data/wiki2/`.

### Quick Start

```bash
cd nanoGPT
python train.py config/train_wiki2_small.py
```

This runs a 6-layer, 4-head, 128-dim transformer on WikiText-2 for 100k steps, evaluating every 1000 steps.

To override individual settings on the command line:

```bash
python train.py config/train_wiki2_small.py \
    --max_iters=50000 \
    --device=cpu \
    --eval_interval=500
```

### Configuration

The config file is `nanoGPT/config/train_wiki2_small.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layer` | 6 | Number of transformer layers |
| `n_head` | 4 | Number of attention heads |
| `n_embd` | 128 | Embedding dimension |
| `block_size` | 32 | Context length (matches LUT default) |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 16 | Batch size |
| `max_iters` | 100000 | Training steps |
| `learning_rate` | 3e-4 | Peak Adam learning rate |
| `warmup_iters` | 4000 | LR warmup steps |
| `eval_interval` | 1000 | Steps between evaluations |
| `device` | `mps` | `cpu`, `cuda`, or `mps` (Apple Silicon) |

**Generating text from a saved checkpoint:**

```bash
cd nanoGPT
python sample.py --out_dir=out-wiki2-small --start="The history of"
```

---

## Comparing the Two Models

Both models are trained on the same WikiText-2 data with GPT-2 tokenization (vocab size 50,257) and the same context length (32 tokens by default). Loss is measured as average negative log-likelihood; perplexity = exp(loss).

| Aspect | LUT model | nanoGPT transformer |
|--------|-----------|---------------------|
| Computation | Discrete table lookups (no matmul) | Dense matrix multiplications |
| Attention | Binary-comparison indexed LUTs | Softmax over dot products |
| Optimizer | Gradient descent (custom LR schedule) | AdamW |
| Hardware | CPU-optimized (Numba + prange) | GPU-friendly (PyTorch) |
| Config entry | `main.py --embedding-dim 32 ...` | `nanoGPT/config/train_wiki2_small.py` |
| Loss log | `loss.csv` | stdout + optional wandb |

For a fair speed comparison, run both on the same hardware. The LUT model is optimized for CPU with Numba parallel kernels; the transformer benefits significantly from a GPU.
