#!/usr/bin/env python3
"""Inspect train.bin tokens next to the existing POS cache labels.

This script is intentionally read-only with respect to the dataset artifacts: it
loads train.bin, decodes those token ids through the training vocabulary, loads
the already-created POS .npy cache used by nanoGPT_POS/train.py, and prints both
streams by shared index.

It does not run spaCy and it does not create/rebuild the POS cache. If the cache
is missing or has the wrong length, the script fails so pipeline problems stay
visible.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
NANOGPT_POS_DIR = ROOT_DIR / "nanoGPT_POS"
DATA_DIR = NANOGPT_POS_DIR / "data" / "wiki2"
DEFAULT_TXT = DATA_DIR / "train.txt"
DEFAULT_BIN = DATA_DIR / "train.bin"
DEFAULT_VOCAB = ROOT_DIR / "vocab_files" / "20_freq_wiki2_word_tok.txt"
DEFAULT_POS_CACHE_DIR = ROOT_DIR / ".pos_cache"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from word_tokenizer import POS_LABELS, WordTokenizer  # noqa: E402


def token_stream_hash(token_ids: np.ndarray) -> str:
    tokens = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha256(tokens.tobytes()).hexdigest()[:16]


def pos_cache_path(input_path: Path, token_ids: np.ndarray, vocab_file: Path, cache_dir: Path) -> Path:
    """Use the token-stream POS cache naming logic from main.py/train.py."""
    token_hash = token_stream_hash(token_ids)
    cache_name = f"{input_path.name}.{vocab_file.name}.tokenized.{token_hash}.pos.npy"
    return cache_dir / cache_name


def load_existing_pos_cache(txt_path: Path, token_ids: np.ndarray, vocab_file: Path,
                            cache_dir: Path) -> np.ndarray:
    token_count = len(token_ids)
    cache_path = pos_cache_path(txt_path, token_ids, vocab_file, cache_dir)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"POS cache not found: {cache_path}\n"
            "Run the training/preparation path that creates pos_cache first. "
            "This inspector will not build POS labels for you."
        )

    pos_ids = np.load(cache_path)
    if len(pos_ids) != token_count:
        raise ValueError(
            f"POS cache length mismatch for {cache_path}: "
            f"{len(pos_ids)} POS labels vs {token_count} train.bin tokens"
        )

    print(f"Loaded train.bin : {token_count:,} token ids", file=sys.stderr)
    print(f"Loaded POS cache : {cache_path}", file=sys.stderr)
    print(f"Loaded POS labels: {len(pos_ids):,} labels", file=sys.stderr)
    return pos_ids.astype(np.int64, copy=False)


def decode_token_ids(token_ids: np.ndarray, tokenizer: WordTokenizer) -> list[str]:
    tokens = []
    for token_id in token_ids:
        idx = int(token_id)
        if 0 <= idx < len(tokenizer.id_to_token):
            tokens.append(tokenizer.id_to_token[idx])
        else:
            tokens.append("<bad-token-id>")
    return tokens


def pos_label(pos_id: int) -> str:
    if 0 <= pos_id < len(POS_LABELS):
        return POS_LABELS[pos_id]
    return f"<bad-pos-id:{pos_id}>"


def print_pairs(tokens: list[str], pos_ids: np.ndarray, limit: int | None, skip_eol: bool) -> None:
    rows: list[tuple[int, str, int, str]] = []
    for idx, (token, pos_id) in enumerate(zip(tokens, pos_ids)):
        if skip_eol and token == "<eol>":
            continue
        pos_id_int = int(pos_id)
        rows.append((idx, token, pos_id_int, pos_label(pos_id_int)))
        if limit is not None and len(rows) >= limit:
            break

    index_width = max(len("IDX"), *(len(str(idx)) for idx, _, _, _ in rows)) if rows else len("IDX")
    token_width = max(len("TRAIN.BIN_TOKEN"), *(len(token) for _, token, _, _ in rows)) if rows else len("TRAIN.BIN_TOKEN")
    pos_id_width = max(len("POS_ID"), *(len(str(pos_id)) for _, _, pos_id, _ in rows)) if rows else len("POS_ID")
    pos_width = max(len("POS_LABEL"), *(len(pos) for _, _, _, pos in rows)) if rows else len("POS_LABEL")

    print(
        f"{'IDX':>{index_width}}  "
        f"{'TRAIN.BIN_TOKEN':<{token_width}}  "
        f"{'POS_ID':>{pos_id_width}}  "
        f"{'POS_LABEL':<{pos_width}}"
    )
    print(
        f"{'-' * index_width}  "
        f"{'-' * token_width}  "
        f"{'-' * pos_id_width}  "
        f"{'-' * pos_width}"
    )
    for idx, token, pos_id, label in rows:
        print(f"{idx:>{index_width}}  {token:<{token_width}}  {pos_id:>{pos_id_width}}  {label:<{pos_width}}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show train.bin decoded tokens next to existing POS cache labels."
    )
    parser.add_argument("--txt", type=Path, default=DEFAULT_TXT, help="Path to train.txt used for the POS cache hash")
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="Path to train.bin")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB, help="Word vocabulary used to create train.bin")
    parser.add_argument(
        "--pos-cache-dir",
        type=Path,
        default=DEFAULT_POS_CACHE_DIR,
        help="Directory containing train.py POS .npy caches",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum rows to print. Use 0 to print all.",
    )
    parser.add_argument(
        "--skip-eol",
        action="store_true",
        help="Hide <eol> tokens from the printed table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_ids = np.memmap(args.bin, dtype=np.uint16, mode="r")
    tokenizer = WordTokenizer(args.vocab)
    pos_ids = load_existing_pos_cache(
        txt_path=args.txt,
        token_ids=token_ids,
        vocab_file=args.vocab,
        cache_dir=args.pos_cache_dir,
    )

    tokens = tokenizer.tokens_from_ids(token_ids)
    limit = None if args.limit == 0 else args.limit
    print_pairs(tokens, pos_ids, limit=limit, skip_eol=args.skip_eol)


if __name__ == "__main__":
    main()
