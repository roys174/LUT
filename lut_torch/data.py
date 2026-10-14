"""Data loading, POS caching and batched snippet sampling.

Reuses ``word_tokenizer.WordTokenizer`` and reproduces ``main.py``'s POS cache path
verbatim, so an existing ``.pos_cache`` directory (e.g. ``nanoGPT_POS/.pos_cache``) is
picked up rather than rebuilt through spaCy.

The one structural change from ``main.py`` is batching: instead of one snippet per step
plus a multiprocessing replay of averaged sparse gradients, snippets are stacked into a
real leading batch dimension.
"""

import hashlib
import json
import os
import sys

import numpy as np

# Kept in sync with main.py.
UNK_TOKEN_ID = 0
IGNORE_INDEX = -1


# Pre-tokenized streams (written by prepare_wikitext103.py): a raw little-endian token
# file `X.bin` plus `X.bin.json`. Only the fields below are read, so this module does not
# import the writer.
TOKEN_BIN_FORMAT = 'lut-token-bin-v1'
_BIN_DTYPES = {'uint16': np.uint16, 'uint32': np.uint32, 'int32': np.int32}

_WHITESPACE_BYTES = b' \n\t\r\x0b\x0c'


def read_token_bin_manifest(bin_path):
    """Validated manifest of a ``.bin`` token file; raises ``ValueError`` if it is off."""
    manifest_path = bin_path + '.json'
    if not os.path.exists(manifest_path):
        raise ValueError(f'{bin_path}: missing manifest {manifest_path}')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('format') != TOKEN_BIN_FORMAT:
        raise ValueError(f"{manifest_path}: format {manifest.get('format')!r}, expected "
                         f"{TOKEN_BIN_FORMAT!r}")
    dtype = _BIN_DTYPES.get(manifest.get('dtype'))
    if dtype is None:
        raise ValueError(f"{manifest_path}: unsupported dtype {manifest.get('dtype')!r}")
    expected = int(manifest['n_tokens']) * np.dtype(dtype).itemsize
    actual = os.path.getsize(bin_path)
    if actual != expected:
        raise ValueError(f"{bin_path}: {actual} bytes, but the manifest says "
                         f"{manifest['n_tokens']} {manifest['dtype']} tokens = "
                         f"{expected} bytes")
    return manifest


def load_token_bin(path, tokenizer):
    """Memory-map a ``.bin`` stream after checking its manifest against ``tokenizer``."""
    manifest = read_token_bin_manifest(path)
    if manifest.get('tokenizer') != tokenizer:
        raise ValueError(f"{path} was tokenized with {manifest.get('tokenizer')!r} but "
                         f"--tokenizer is {tokenizer!r}")
    tokens = np.memmap(path, dtype=_BIN_DTYPES[manifest['dtype']], mode='r')
    print(f"  {path}: {len(tokens):,} tokens, {manifest.get('n_words')} words, "
          f"sha256 {str(manifest.get('tokens_sha256'))[:12]}")
    return tokens


class WordTables:
    """Per-vocab byte facts that decide whether a BPE token starts a whitespace word.

    ``has_nonws[i]``: token ``i``'s bytes contain a non-whitespace byte; ``starts_ws[i]`` /
    ``ends_ws[i]``: its first / last byte is whitespace. Built once from the tokenizer.
    """

    def __init__(self, enc):
        n = enc.n_vocab
        self.has_nonws = np.zeros(n, dtype=bool)
        self.starts_ws = np.zeros(n, dtype=bool)
        self.ends_ws = np.zeros(n, dtype=bool)
        for i in range(n):
            try:
                b = enc.decode_single_token_bytes(i)
            except KeyError:        # unassigned id in a padded vocabulary
                continue
            if not b:
                continue
            self.has_nonws[i] = any(c not in _WHITESPACE_BYTES for c in b)
            self.starts_ws[i] = b[0] in _WHITESPACE_BYTES
            self.ends_ws[i] = b[-1] in _WHITESPACE_BYTES

    def starts_word(self, prev, cur):
        """Boolean mask: does token ``cur`` begin a ``str.split()`` word after ``prev``?

        ``prev`` is the preceding token id, or -1 at the start of the stream (which counts
        as following whitespace). Arrays of equal shape.
        """
        prev = np.asarray(prev)
        cur = np.asarray(cur)
        prev_ws = np.where(prev < 0, True, self.ends_ws[np.maximum(prev, 0)])
        return self.has_nonws[cur] & (self.starts_ws[cur] | prev_ws)


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

    def __init__(self, tokens, pos_data, context_size, pos_task,
                 unk_id=UNK_TOKEN_ID, word_tables=None):
        # Kept in its own dtype (int32 from text, a uint16 memmap from .bin): a 100M-token
        # stream as int64 is gigabytes. Only the gathered window is widened, in batch().
        self.tokens = np.asarray(tokens)
        # What evaluate() needs to know about the tokenizer: the id that means <unk>
        # (None when the vocabulary has none) and the BPE word-start tables (None for
        # the word tokenizer, whose tokens already are words).
        self.unk_id = unk_id
        self.word_tables = word_tables
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
        window = self.tokens[offsets].astype(np.int64)      # (B, CS+1)
        inputs = window[:, :cs]

        if self.pos_task:
            targets = self.pos_data[offsets].copy()
            # main.py masks POS targets wherever the *input* token is <unk>.
            targets[window == UNK_TOKEN_ID] = IGNORE_INDEX
            targets = targets[:, :cs]
        else:
            targets = window[:, 1:cs + 1]
        return inputs, targets


def _is_bin(path):
    return bool(path) and path.endswith('.bin')


def load_datasets(args, enc):
    """Load train/validation streams and the fixed validation start offsets.

    A ``.bin`` path is a pre-tokenized stream (see ``read_token_bin_manifest``); both
    splits must then be ``.bin``, and POS is unavailable because it needs the text.
    """
    bin_input = _is_bin(args.training_data)
    if args.validation_data and _is_bin(args.validation_data) != bin_input:
        raise ValueError('training and validation data must both be .bin or both text')
    if bin_input and args.POS_task:
        raise ValueError('--POS-task needs text input; .bin streams carry no POS labels')

    if bin_input:
        print('Pre-tokenized input:')
        train_tokens = load_token_bin(args.training_data, args.tokenizer)
    else:
        with open(args.training_data, 'r', encoding='utf-8') as f:
            train_tokens = np.array(enc.encode(f.read()), dtype=np.int32)
    train_pos = (load_or_build_pos_encoding(args.training_data, train_tokens, enc, args,
                                            'training data')
                 if args.POS_task else None)
    print(f"Training data: {len(train_tokens)} tokens from {args.training_data}")

    if args.validation_data:
        if bin_input:
            val_tokens = load_token_bin(args.validation_data, args.tokenizer)
        else:
            with open(args.validation_data, 'r', encoding='utf-8') as f:
                val_tokens = np.array(enc.encode(f.read()), dtype=np.int32)
        val_pos = (load_or_build_pos_encoding(args.validation_data, val_tokens, enc,
                                              args, 'validation data')
                   if args.POS_task else None)
        print(f"Validation data: {len(val_tokens)} tokens from {args.validation_data}")
    else:
        val_tokens, val_pos = train_tokens, train_pos
        print("Validation data: sampled from training data")

    # Token id 0 is <unk> only in the word vocabulary; under BPE it is a real token ("!"
    # in GPT-2). BPE streams also get the word-start tables for per-word loss.
    is_word = args.tokenizer == 'word'
    unk_id = UNK_TOKEN_ID if is_word else None
    tables = None if is_word else WordTables(enc)
    train = SnippetDataset(train_tokens, train_pos, args.context_size, args.POS_task,
                           unk_id=unk_id, word_tables=tables)
    val = SnippetDataset(val_tokens, val_pos, args.context_size, args.POS_task,
                         unk_id=unk_id, word_tables=tables)
    return train, val
