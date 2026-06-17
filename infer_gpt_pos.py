#!/usr/bin/env python3
"""Interactive POS inference for saved nanoGPT POS checkpoints."""

import argparse
import contextlib
import os
import sys

import torch

from pos_ambiguity_examples import AMBIGUITY_EXAMPLES
from word_tokenizer import POS_LABELS, WordTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.join(SCRIPT_DIR, "nanoGPT_POS")
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)

from model import GPT, GPTConfig  # noqa: E402

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


def strip_compile_prefix(state_dict):
    unwanted_prefix = "_orig_mod."
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(unwanted_prefix):
            key = key[len(unwanted_prefix):]
        stripped[key] = value
    return stripped


def choose_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def load_gpt_pos(checkpoint_dir, device_arg):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    ckpt_path = os.path.join(checkpoint_dir, "ckpt.pt")
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Expected a checkpoint directory, got: {checkpoint_dir}")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Missing checkpoint file: {ckpt_path}")

    device = choose_device(device_arg)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = dict(checkpoint["model_args"])
    checkpoint_config = dict(checkpoint.get("config", {}))

    state_dict = strip_compile_prefix(checkpoint["model"])
    state_output = state_dict.get("lm_head.weight")
    vocab_size_output = model_args.pop("vocab_size_output", None)
    if state_output is not None:
        vocab_size_output = int(state_output.shape[0])
    if vocab_size_output is None:
        vocab_size_output = len(POS_LABELS)

    config_keys = set(GPTConfig.__dataclass_fields__)
    gpt_args = {key: value for key, value in model_args.items() if key in config_keys}
    gptconf = GPTConfig(**gpt_args)
    with contextlib.redirect_stdout(sys.stderr):
        model = GPT(gptconf)
    model.lm_head = torch.nn.Linear(model.config.n_embd, vocab_size_output, bias=False)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    vocab_file = checkpoint_config.get("vocab_file")
    if not vocab_file:
        raise ValueError("Checkpoint config does not contain vocab_file.")
    bases = [
        os.getcwd(),
        checkpoint_config.get("script_dir"),
        checkpoint_config.get("project_dir"),
        NANOGPT_DIR,
        SCRIPT_DIR,
        checkpoint_dir,
    ]
    vocab_file = resolve_existing_path(vocab_file, bases)
    if not os.path.exists(vocab_file):
        raise ValueError(f"Could not resolve vocab_file from checkpoint config: {vocab_file}")

    tokenizer = WordTokenizer(vocab_file)
    return model, tokenizer, device


def predict_pos(model, tokenizer, device, text):
    token_ids = tokenizer.encode(text)
    token_text = tokenizer.tokens_from_ids(token_ids)
    labels = []
    block_size = model.config.block_size

    with torch.no_grad():
        for start in range(0, len(token_ids), block_size):
            chunk = token_ids[start:start + block_size]
            if not chunk:
                continue
            idx = torch.tensor([chunk], dtype=torch.long, device=device)
            dummy_targets = torch.zeros_like(idx)
            logits, _ = model(idx, dummy_targets)
            preds = torch.argmax(logits[0], dim=-1).detach().cpu().tolist()
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


def run_ambiguity_examples(model, tokenizer, device, output_dir=None):
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
            _, labels = predict_pos(model, tokenizer, device, text)
            predicted = target_prediction(tokenizer, text, target, labels)
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


def run_repl(model, tokenizer, device):
    print("Enter text to POS-tag. Blank line, :q, or :quit exits.", file=sys.stderr)
    while True:
        try:
            text = input("text> ")
        except EOFError:
            print(file=sys.stderr)
            return
        if text == "" or text in {":q", ":quit"}:
            return
        tokens, labels = predict_pos(model, tokenizer, device, text)
        print_table(tokens, labels)


def main():
    parser = argparse.ArgumentParser(description="Infer POS tags with a saved GPT_POS checkpoint.")
    parser.add_argument("checkpoint_dir", help="Directory containing ckpt.pt")
    parser.add_argument("--once", help="Tag this text once and exit")
    parser.add_argument("--infer-ambiguate", action="store_true",
                        help="Run the built-in ambiguous-word POS examples and exit")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    try:
        model, tokenizer, device = load_gpt_pos(args.checkpoint_dir, args.device)
        if args.infer_ambiguate:
            output_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
            run_ambiguity_examples(model, tokenizer, device, output_dir)
        elif args.once is not None:
            tokens, labels = predict_pos(model, tokenizer, device, args.once)
            print_table(tokens, labels)
        else:
            run_repl(model, tokenizer, device)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
