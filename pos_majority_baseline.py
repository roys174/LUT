#!/usr/bin/env python3

"""Majority-POS baseline for cached POS-tagged wiki2 data.

This script mirrors the POS cache naming and length validation used by main.py,
then evaluates a per-word empirical POS classifier on train/validation text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys

import numpy as np

from word_tokenizer import POS_LABELS, WordTokenizer


IGNORE_INDEX = -1
UNK_TOKEN_ID = 0


def token_stream_hash(token_ids):
    tokens = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha256(tokens.tobytes()).hexdigest()[:16]


def pos_cache_path(input_path, token_ids, args):
    token_hash = token_stream_hash(token_ids)
    vocab_id = os.path.basename(args.vocab_file) if args.tokenizer == "word" else args.tokenizer
    cache_name = f"{os.path.basename(input_path)}.{vocab_id}.tokenized.{token_hash}.pos.npy"
    return os.path.join(args.pos_cache_dir, cache_name)


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise SystemExit(f"error opening text file {path}: {exc}") from exc


def load_pos_cache(input_path, token_ids, args, name):
    cache_path = pos_cache_path(input_path, token_ids, args)
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"missing POS cache for {name}: {cache_path}\n"
            f"run main.py once with the same text and vocab to build it"
        )

    pos_data = np.load(cache_path)
    if len(pos_data) != len(token_ids):
        raise SystemExit(
            f"stale POS cache for {name}: expected {len(token_ids)} labels, "
            f"found {len(pos_data)} in {cache_path}"
        )

    print(f"loaded POS cache for {name}: {cache_path}")
    return pos_data.astype(np.int32, copy=False)


def valid_mask(token_ids):
    return np.asarray(token_ids, dtype=np.int32) != UNK_TOKEN_ID


def summarize_distribution(counts, smoothing):
    smoothed = counts.astype(np.float64) + smoothing
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def predict_and_score(train_tokens, train_pos, val_tokens, val_pos, smoothing):
    train_max = int(train_tokens.max()) if train_tokens.size else -1
    val_max = int(val_tokens.max()) if val_tokens.size else -1
    num_words = max(train_max, val_max) + 1
    num_pos_labels = len(POS_LABELS)

    train_valid = valid_mask(train_tokens)

    counts = np.zeros((num_words, num_pos_labels), dtype=np.int64)
    np.add.at(counts, (train_tokens[train_valid], train_pos[train_valid]), 1)

    global_counts = np.bincount(train_pos[train_valid], minlength=num_pos_labels).astype(np.int64)
    global_dist = (global_counts.astype(np.float64) + smoothing)
    global_dist /= global_dist.sum()

    word_dist = summarize_distribution(counts, smoothing)
    word_counts = counts.sum(axis=1)
    fallback_rows = word_counts == 0
    if fallback_rows.any():
        word_dist[fallback_rows] = global_dist

    val_valid = valid_mask(val_tokens)
    eval_mask = val_valid
    if not np.any(eval_mask):
        raise SystemExit("no valid validation positions after masking UNK tokens")

    eval_tokens = val_tokens[eval_mask]
    eval_pos = val_pos[eval_mask]

    predicted_pos = np.argmax(word_dist[eval_tokens], axis=1)
    correct = int((predicted_pos == eval_pos).sum())

    probs = word_dist[eval_tokens, eval_pos]
    loss = float(-np.log(probs).mean())
    perplexity = float(math.exp(loss))

    return {
        "accuracy": correct / len(eval_pos),
        "loss": loss,
        "perplexity": perplexity,
        "total": int(len(eval_pos)),
        "correct": correct,
    }


def write_csv(path, metrics):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accuracy", "loss", "perplexity", "total", "correct"])
        writer.writeheader()
        writer.writerow({
            "accuracy": f"{metrics['accuracy']:.12f}",
            "loss": f"{metrics['loss']:.12f}",
            "perplexity": f"{metrics['perplexity']:.12f}",
            "total": metrics["total"],
            "correct": metrics["correct"],
        })


def main():
    parser = argparse.ArgumentParser(description="Majority-POS baseline using cached POS labels")
    parser.add_argument("training_data", type=str, help="Path to training text")
    parser.add_argument("--validation-data", type=str, required=True, help="Path to validation text")
    parser.add_argument("--pos-cache-dir", type=str, default=".pos_cache", help="Directory containing cached .pos.npy files")
    parser.add_argument("--vocab-file", type=str, required=True, help="Vocabulary file for WordTokenizer")
    parser.add_argument("--tokenizer", type=str, default="word", help='Tokenizer name; only "word" is supported')
    parser.add_argument("--smoothing", type=float, default=1e-9, help="Additive smoothing for empirical POS probabilities")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.tokenizer != "word":
        raise SystemExit('this baseline only supports --tokenizer word')

    os.makedirs(args.pos_cache_dir, exist_ok=True)

    tokenizer = WordTokenizer(args.vocab_file)

    train_text = read_text(args.training_data)
    val_text = read_text(args.validation_data)

    train_tokens = np.asarray(tokenizer.encode(train_text), dtype=np.int32)
    val_tokens = np.asarray(tokenizer.encode(val_text), dtype=np.int32)

    train_pos = load_pos_cache(args.training_data, train_tokens, args, "training data")
    val_pos = load_pos_cache(args.validation_data, val_tokens, args, "validation data")

    metrics = predict_and_score(train_tokens, train_pos, val_tokens, val_pos, args.smoothing)

    print(f"train tokens: {len(train_tokens)}")
    print(f"validation tokens: {len(val_tokens)}")
    print(f"valid eval positions: {metrics['total']}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"perplexity: {metrics['perplexity']:.6f}")

    if args.output_csv:
        write_csv(args.output_csv, metrics)
        print(f"wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()