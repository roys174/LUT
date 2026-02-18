#!/usr/bin/env python3
"""Pre-tokenize text files into binary format for the C SNN transformer.

Usage:
    python tokenize_data.py input.txt output.bin [--tokenizer gpt2]
    python tokenize_data.py --export-vocab output.vocab [--tokenizer gpt2]

Binary token format (.bin):
    bytes 0-3:   magic 0x4C555442 ("LUTB")
    bytes 4-7:   vocab_size (uint32)
    bytes 8-11:  num_tokens (uint32)
    bytes 12+:   int32[] token IDs

Vocab format (.vocab):
    bytes 0-3:   magic 0x564F4342 ("VOCB")
    bytes 4-7:   vocab_size (uint32)
    bytes 8+:    for each token: uint16 length, then raw bytes
"""

import argparse
import struct
import sys

MAGIC_TOKENS = 0x4C555442  # "LUTB"
MAGIC_VOCAB  = 0x564F4342  # "VOCB"


def tokenize_file(input_path, output_path, tokenizer_name):
    import tiktoken

    enc = tiktoken.get_encoding(tokenizer_name)
    vocab_size = enc.n_vocab

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing with {tokenizer_name} (vocab_size={vocab_size})...")
    tokens = enc.encode(text, allowed_special=set())
    num_tokens = len(tokens)
    print(f"  {len(text)} chars -> {num_tokens} tokens "
          f"(ratio: {len(text)/num_tokens:.2f} chars/token)")

    print(f"Writing {output_path}...")
    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", MAGIC_TOKENS, vocab_size, num_tokens))
        f.write(struct.pack(f"<{num_tokens}i", *tokens))

    file_size = 12 + num_tokens * 4
    print(f"  Wrote {file_size} bytes ({file_size/1024/1024:.1f} MB)")


def export_vocab(output_path, tokenizer_name):
    import tiktoken

    enc = tiktoken.get_encoding(tokenizer_name)
    vocab_size = enc.n_vocab

    print(f"Exporting vocab for {tokenizer_name} (vocab_size={vocab_size})...")

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_VOCAB, vocab_size))
        for i in range(vocab_size):
            try:
                token_bytes = enc.decode_single_token_bytes(i)
            except KeyError:
                token_bytes = b"<?>"
            length = len(token_bytes)
            f.write(struct.pack("<H", length))
            f.write(token_bytes)

    print(f"  Wrote {output_path}")


def verify_roundtrip(bin_path, input_path, tokenizer_name):
    """Verify that decoding the binary file reproduces the original text."""
    import tiktoken

    enc = tiktoken.get_encoding(tokenizer_name)

    with open(bin_path, "rb") as f:
        magic, vocab_size, num_tokens = struct.unpack("<III", f.read(12))
        assert magic == MAGIC_TOKENS, f"Bad magic: {magic:#x}"
        tokens = list(struct.unpack(f"<{num_tokens}i", f.read(num_tokens * 4)))

    decoded = enc.decode(tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        original = f.read()

    if decoded == original:
        print("Round-trip verification: PASSED")
    else:
        # Find first difference
        for i, (a, b) in enumerate(zip(decoded, original)):
            if a != b:
                print(f"Round-trip verification: FAILED at char {i}")
                print(f"  Original: ...{original[max(0,i-20):i+20]!r}...")
                print(f"  Decoded:  ...{decoded[max(0,i-20):i+20]!r}...")
                break
        else:
            print(f"Round-trip verification: FAILED (length mismatch: "
                  f"{len(decoded)} vs {len(original)})")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize text for C SNN transformer")
    parser.add_argument("input", nargs="?", help="Input text file")
    parser.add_argument("output", nargs="?", help="Output binary file")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name (default: gpt2)")
    parser.add_argument("--export-vocab", metavar="PATH", help="Export vocab lookup file")
    parser.add_argument("--verify", action="store_true", help="Verify round-trip after tokenizing")
    args = parser.parse_args()

    if args.export_vocab:
        export_vocab(args.export_vocab, args.tokenizer)

    if args.input and args.output:
        tokenize_file(args.input, args.output, args.tokenizer)
        if args.verify:
            verify_roundtrip(args.output, args.input, args.tokenizer)

    if not args.export_vocab and not (args.input and args.output):
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
