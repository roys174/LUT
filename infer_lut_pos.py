#!/usr/bin/env python3
"""Interactive POS inference for saved LUT POS runs."""

import argparse
import contextlib
import os
import shlex
import sys

import numpy as np

from main import (
    IGNORE_INDEX,
    UNK_TOKEN_ID,
    Model,
    build_Model,
    configure_lut_comparison_mode,
    load_model,
    model_forward,
)
from pos_ambiguity_examples import AMBIGUITY_EXAMPLES
from word_tokenizer import POS_LABELS, WordTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HIGHLIGHT_START = "\033[1;36m"
HIGHLIGHT_END = "\033[0m"



def resolve_existing_path(path, bases):
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    candidates = [os.path.abspath(os.path.join(base, path)) for base in bases if base]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else os.path.abspath(path)


def resolve_model_path(model_arg, run_dir):
    path = os.path.expanduser(model_arg)
    if not os.path.isabs(path):
        path = os.path.join(run_dir, path)
    if os.path.exists(path):
        return path
    if not path.endswith(".npz") and os.path.exists(path + ".npz"):
        return path + ".npz"
    return path


def read_command_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.strip() == "command:":
            for command_line in lines[idx + 1:]:
                command_line = command_line.strip()
                if command_line:
                    return command_line

    for line in lines:
        command_line = line.strip()
        if " main.py " in f" {command_line} ":
            return command_line

    raise ValueError(f"Could not find a saved training command in {config_path}")


def argv_after_main(command_line):
    tokens = shlex.split(command_line)
    for idx, token in enumerate(tokens):
        if os.path.basename(token) == "main.py":
            return tokens[idx + 1:]
    raise ValueError("Saved command is not a main.py LUT training command")


def training_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("training_data")
    parser.add_argument("--validation-data", default=None)
    parser.add_argument("--tokenizer", default="word")
    parser.add_argument("--vocab-file", default="word_vocab.txt")
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--positional-dim", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-t", type=int, default=16)
    parser.add_argument("--n-c", type=int, default=6)
    parser.add_argument("--lut-comparison-mode", choices=["random", "no-overlap", "overlap"],
                        default="random")
    parser.add_argument("--attn_qk_dim", type=int, default=8)
    parser.add_argument("--attn_qk_n_c", type=int, default=3)
    parser.add_argument("--attn_qk_n_t", type=int, default=4)
    parser.add_argument("--testing-length", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=100000000)
    parser.add_argument("--validation-interval", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--loss-file", default="loss.csv")
    parser.add_argument("--factored-output", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-cores", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--numba-threads", type=int, default=0)
    parser.add_argument("--replay-workers", type=int, default=0)
    parser.add_argument("--soft-attention", action="store_true")
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--lut-update-stats-file", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lut")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-log-interval", type=int, default=100)
    parser.add_argument("--POS-task", "--pos-task", dest="POS_task", action="store_true")
    parser.add_argument("--pos-cache-dir", default=".pos_cache")
    parser.add_argument("--new-attention", action="store_true")
    parser.add_argument("--PE-only", dest="PE_only", action="store_true")
    parser.add_argument("--qkpe-direct-sign", action="store_true")
    return parser


def load_training_args(run_dir):
    config_path = os.path.join(run_dir, "config.txt")
    if not os.path.exists(config_path):
        seed_dirs = [
            name for name in os.listdir(run_dir)
            if name.startswith("seed_") and os.path.isdir(os.path.join(run_dir, name))
        ] if os.path.isdir(run_dir) else []
        if seed_dirs:
            raise ValueError(
                "This looks like a multiseed parent directory. "
                "Pass one seed directory, for example: "
                f"{os.path.join(run_dir, sorted(seed_dirs)[0])}"
            )
        raise ValueError(f"Missing config.txt in {run_dir}")

    command_line = read_command_from_config(config_path)
    command_argv = argv_after_main(command_line)
    parser = training_arg_parser()
    parsed, unknown = parser.parse_known_args(command_argv)
    if unknown:
        print(f"Warning: ignored unrecognized training flags: {' '.join(unknown)}", file=sys.stderr)
    configure_lut_comparison_mode(parsed, parser)

    if not parsed.POS_task:
        raise ValueError("The saved LUT command is not a POS-task run.")
    if parsed.tokenizer != "word":
        raise ValueError("LUT POS inference currently requires --tokenizer word.")
    if parsed.factored_output:
        raise ValueError("POS checkpoints should not use --factored-output.")

    bases = [os.getcwd(), SCRIPT_DIR, run_dir]
    parsed.training_data = resolve_existing_path(parsed.training_data, bases)
    parsed.vocab_file = resolve_existing_path(parsed.vocab_file, bases)

    enc = WordTokenizer(parsed.vocab_file)
    parsed.vocab_size_input = enc.n_vocab
    parsed.vocab_size_output = len(POS_LABELS)
    parsed.enc = enc
    parsed.vocab_hi = (parsed.vocab_size_output + 255) // 256
    parsed.vocab_lo = 256
    parsed.pe_bitmasks = (1 << np.arange(parsed.positional_dim)).astype(np.int32)
    parsed.shift_qk = parsed.n_c + parsed.positional_dim
    return parsed


def load_lut_model(run_dir, model_arg):
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    if os.path.isdir(run_dir):
        seed_dirs = [
            name for name in os.listdir(run_dir)
            if name.startswith("seed_") and os.path.isdir(os.path.join(run_dir, name))
        ]
        if seed_dirs and not os.path.exists(os.path.join(run_dir, "model.npz")):
            raise ValueError(
                "This looks like a multiseed parent directory. "
                "Pass one seed directory, for example: "
                f"{os.path.join(run_dir, sorted(seed_dirs)[0])}"
            )

    args = load_training_args(run_dir)
    model_path = resolve_model_path(model_arg, run_dir)
    if not os.path.exists(model_path):
        raise ValueError(f"Missing model file: {model_path}")

    model = Model(args)
    build_Model(model, args)
    with contextlib.redirect_stdout(sys.stderr):
        load_model(model, model_path, args)
    return model, args


def predict_pos(model, args, text):
    token_ids = args.enc.encode(text)
    token_text = args.enc.tokens_from_ids(token_ids)
    labels = []
    context_size = args.context_size

    for start in range(0, len(token_ids), context_size):
        chunk = token_ids[start:start + context_size]
        if not chunk:
            continue
        n = len(chunk)
        model.tokens[:] = UNK_TOKEN_ID
        model.tokens[:n] = np.asarray(chunk, dtype=np.int32)
        model.z[:] = model.Token_embedder[model.tokens[:context_size]]
        model.output_head.load_targets(
            np.full(context_size + 1, IGNORE_INDEX, dtype=np.int32),
            context_size,
        )
        model_forward(model, args, training=False)
        preds = np.argmax(model.output_head.output[:n], axis=1).astype(int)
        labels.extend(POS_LABELS[p] if 0 <= p < len(POS_LABELS) else "X" for p in preds)

    return token_text, labels


def print_table(tokens, labels):
    print("token\tpredicted_POS")
    for token, label in zip(tokens, labels):
        print(f"{token}\t{label}")


def highlight_sentence(sentence, target):
    highlighted = f"{HIGHLIGHT_START}{target}{HIGHLIGHT_END}"
    return sentence.replace(target, highlighted, 1)


def target_prediction(tokenizer, text, target, labels):
    raw_tokens = tokenizer._split(text)
    target_index = None
    for idx, token in enumerate(raw_tokens):
        if token == target:
            target_index = idx
            break
    if target_index is None:
        for idx, token in enumerate(raw_tokens):
            if token.lower() == target.lower():
                target_index = idx
                break
    if target_index is None or target_index >= len(labels):
        return "?"
    return labels[target_index]


def label_matches(predicted, expected):
    return predicted in expected.split("/")


def run_ambiguity_examples(model, model_args, output_dir=None):
    terminal_lines = ["Ambiguous POS examples"]
    file_lines = ["Ambiguous POS examples"]
    correct = 0
    total = 0
    at_least_one_words = 0
    at_least_one_correct = 0
    at_least_one_total = 0
    both_correct_words = 0
    for word, examples in AMBIGUITY_EXAMPLES:
        terminal_lines.append(f"\n{word}")
        file_lines.append(f"\n{word}")
        word_correct = 0
        word_total = 0
        for text, target, expected in examples:
            _, labels = predict_pos(model, model_args, text)
            predicted = target_prediction(model_args.enc, text, target, labels)
            is_correct = label_matches(predicted, expected)
            correct += int(is_correct)
            total += 1
            word_correct += int(is_correct)
            word_total += 1
            status = "OK" if is_correct else "MISS"
            terminal_lines.append(f"  {highlight_sentence(text, target)}")
            file_lines.append(f"  {text}")
            result = f"    expected: {expected:<7} predicted: {predicted:<5} {status}"
            terminal_lines.append(result)
            file_lines.append(result)
        if word_correct > 0:
            at_least_one_words += 1
            at_least_one_correct += word_correct
            at_least_one_total += word_total
            if word_correct == word_total:
                both_correct_words += 1
    accuracy = correct / total if total else 0.0
    summary = f"\nSummary: {correct}/{total} correct ({accuracy:.1%})"
    if at_least_one_total:
        at_least_one_accuracy = at_least_one_correct / at_least_one_total
    else:
        at_least_one_accuracy = 0.0
    at_least_one_summary = (
        "At-least-one-correct words: "
        f"{at_least_one_words}/{len(AMBIGUITY_EXAMPLES)} words; "
        "sentence accuracy within those words: "
        f"{at_least_one_correct}/{at_least_one_total} ({at_least_one_accuracy:.1%})"
    )
    if at_least_one_words:
        both_correct_rate = both_correct_words / at_least_one_words
    else:
        both_correct_rate = 0.0
    both_correct_summary = (
        "Both-correct among at-least-one words: "
        f"{both_correct_words}/{at_least_one_words} words ({both_correct_rate:.1%})"
    )
    terminal_lines.append(summary)
    terminal_lines.append(at_least_one_summary)
    terminal_lines.append(both_correct_summary)
    file_lines.append(summary)
    file_lines.append(at_least_one_summary)
    file_lines.append(both_correct_summary)

    print("\n".join(terminal_lines))
    if output_dir is not None:
        output_path = os.path.join(output_dir, "ambiguate_results.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(file_lines) + "\n")
        print(f"Saved ambiguity results to {output_path}", file=sys.stderr)


def run_repl(model, model_args):
    print("Enter text to POS-tag. Blank line, :q, or :quit exits.", file=sys.stderr)
    while True:
        try:
            text = input("text> ")
        except EOFError:
            print(file=sys.stderr)
            return
        if text == "" or text in {":q", ":quit"}:
            return
        tokens, labels = predict_pos(model, model_args, text)
        print_table(tokens, labels)


def main():
    parser = argparse.ArgumentParser(description="Infer POS tags with a saved LUT POS run.")
    parser.add_argument("run_dir", help="Run directory containing config.txt and model.npz")
    parser.add_argument("--once", help="Tag this text once and exit")
    parser.add_argument("--infer-ambiguate", action="store_true",
                        help="Run the built-in ambiguous-word POS examples and exit")
    parser.add_argument("--model", default="model.npz", help="Model file name/path (default: model.npz)")
    args = parser.parse_args()

    try:
        model, model_args = load_lut_model(args.run_dir, args.model)
        if args.infer_ambiguate:
            output_dir = os.path.abspath(os.path.expanduser(args.run_dir))
            run_ambiguity_examples(model, model_args, output_dir)
        elif args.once is not None:
            tokens, labels = predict_pos(model, model_args, args.once)
            print_table(tokens, labels)
        else:
            run_repl(model, model_args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
