"""Data loading, POS caching and batched snippet sampling.

Reuses ``word_tokenizer.WordTokenizer`` and reproduces ``main.py``'s POS cache path
verbatim, so an existing ``.pos_cache`` directory (e.g. ``nanoGPT_POS/.pos_cache``) is
picked up rather than rebuilt through spaCy.

The one structural change from ``main.py`` is batching: instead of one snippet per step
plus a multiprocessing replay of averaged sparse gradients, snippets are stacked into a
real leading batch dimension.
"""

import hashlib
import os
import sys

import numpy as np

# Kept in sync with main.py.
UNK_TOKEN_ID = 0
IGNORE_INDEX = -1


def token_stream_hash(token_ids):
    tokens = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha256(tokens.tobytes()).hexdigest()[:16]


def pos_cache_path(input_path, token_ids, cache_dir, tokenizer, vocab_file):
    """Same filename scheme as ``main.pos_cache_path`` so caches are shared."""
    token_hash = token_stream_hash(token_ids)
    vocab_id = os.path.basename(vocab_file) if tokenizer == 'word' else tokenizer
    name = (f"{os.path.basename(input_path)}.{vocab_id}.tokenized."
            f"{token_hash}.pos.npy")
    return os.path.join(cache_dir, name)


def load_or_build_pos_encoding(input_path, token_ids, enc, args, name):
    os.makedirs(args.pos_cache_dir, exist_ok=True)
    cache_path = pos_cache_path(input_path, token_ids, args.pos_cache_dir,
                                args.tokenizer, args.vocab_file)
    token_count = len(token_ids)

    if os.path.exists(cache_path):
        pos_data = np.load(cache_path)
        if len(pos_data) == token_count:
            print(f"Loaded POS encoding for {name}: {cache_path}")
            return pos_data.astype(np.int32, copy=False)
        print(f"Ignoring stale POS cache for {name}: expected {token_count} labels, "
              f"found {len(pos_data)}")

    print(f"Building POS encoding for {name} from tokenized input...")
    pos_data = np.array(enc.encode_pos_ids(token_ids), dtype=np.int32)
    if len(pos_data) != token_count:
        print(f"POS encoding length mismatch for {name}: {len(pos_data)} POS labels "
              f"vs {token_count} tokens")
        sys.exit(1)
    np.save(cache_path, pos_data)
    print(f"Saved POS encoding for {name}: {cache_path}")
    return pos_data


class SnippetDataset:
    """A token stream plus optional POS labels, sampled as fixed-length windows."""

    def __init__(self, tokens, pos_data, context_size, pos_task):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.pos_data = None if pos_data is None else np.asarray(pos_data, dtype=np.int64)
        self.context_size = context_size
        self.pos_task = pos_task
        self.length = len(self.tokens) - context_size - 1
        if self.length < 1:
            raise ValueError(
                f'need at least {context_size + 2} tokens, got {len(self.tokens)}')

    def batch(self, starts):
        """Return ``(inputs, targets)`` for the given start offsets.

        ``inputs`` is ``(B, CS)``; ``targets`` is ``(B, CS)`` -- the POS label at each
        position, or the next token, with ``IGNORE_INDEX`` where the label is masked.
        """
        cs = self.context_size
        offsets = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(cs + 1)
        window = self.tokens[offsets]                       # (B, CS+1)
        inputs = window[:, :cs]

        if self.pos_task:
            targets = self.pos_data[offsets].copy()
            # main.py masks POS targets wherever the *input* token is <unk>.
            targets[window == UNK_TOKEN_ID] = IGNORE_INDEX
            targets = targets[:, :cs]
        else:
            targets = window[:, 1:cs + 1]
        return inputs, targets


def load_datasets(args, enc):
    """Load train/validation streams and the fixed validation start offsets."""
    with open(args.training_data, 'r', encoding='utf-8') as f:
        train_tokens = np.array(enc.encode(f.read()), dtype=np.int32)
    train_pos = (load_or_build_pos_encoding(args.training_data, train_tokens, enc, args,
                                            'training data')
                 if args.POS_task else None)
    print(f"Training data: {len(train_tokens)} tokens from {args.training_data}")

    if args.validation_data:
        with open(args.validation_data, 'r', encoding='utf-8') as f:
            val_tokens = np.array(enc.encode(f.read()), dtype=np.int32)
        val_pos = (load_or_build_pos_encoding(args.validation_data, val_tokens, enc,
                                              args, 'validation data')
                   if args.POS_task else None)
        print(f"Validation data: {len(val_tokens)} tokens from {args.validation_data}")
    else:
        val_tokens, val_pos = train_tokens, train_pos
        print("Validation data: sampled from training data")

    train = SnippetDataset(train_tokens, train_pos, args.context_size, args.POS_task)
    val = SnippetDataset(val_tokens, val_pos, args.context_size, args.POS_task)
    return train, val
