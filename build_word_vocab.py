"""Build the word-tokenizer vocabulary from a training text file.

Usage:
    python build_word_vocab.py wiki2_train.txt
    python build_word_vocab.py wiki2_train.txt --min-freq 5 --output my_vocab.txt
"""

import argparse
from word_tokenizer import build_vocab

parser = argparse.ArgumentParser(description='Build word tokenizer vocabulary')
parser.add_argument('training_data', type=str, help='Path to training text file')
parser.add_argument('--min-freq', type=int, default=3,
                    help='Minimum word frequency to include in vocabulary (default: 3)')
parser.add_argument('--output', type=str, default='word_vocab.txt',
                    help='Output vocabulary file path (default: word_vocab.txt)')
args = parser.parse_args()

with open(args.training_data, 'r', encoding='utf-8') as f:
    text = f.read()

build_vocab(text, min_freq=args.min_freq, vocab_path=args.output)
