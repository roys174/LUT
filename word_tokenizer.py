"""Word-based tokenizer for the LUT model.

Vocabulary is built from training text, keeping words that appear >= min_freq
times. Punctuation is split into separate tokens. Case-sensitive.

Special tokens:
  <unk>  (ID 0) — rare or unseen words
  <eol>  (ID 1) — end of line

Usage:
  # Build vocab from training text (run once):
  python build_word_vocab.py wiki2_train.txt

  # Then train:
  python main.py wiki2_train.txt --tokenizer word --vocab-file word_vocab.txt
"""

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

UNK = '<unk>'
EOL = '<eol>'


class WordTokenizer:
    """Duck-types the tiktoken encoder interface used by main.py."""

    def __init__(self, vocab_path):
        self.id_to_token = []
        self.token_to_id = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.rstrip('\n')
                self.id_to_token.append(token)
                self.token_to_id[token] = idx
        assert self.id_to_token[0] == UNK, f"First vocab entry must be {UNK}"
        assert self.id_to_token[1] == EOL, f"Second vocab entry must be {EOL}"

    @property
    def n_vocab(self):
        return len(self.id_to_token)

    def _split(self, text):
        """Split text into raw string tokens, inserting <eol> at each line end."""
        tokens = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            tokens.extend(_TOKEN_RE.findall(line))
            if i < len(lines) - 1 or line:
                tokens.append(EOL)
        return tokens

    def encode(self, text):
        return [self.token_to_id.get(t, 0) for t in self._split(text)]

    def decode(self, ids):
        """Convert token IDs to a human-readable string."""
        parts = [self.id_to_token[i] if i < len(self.id_to_token) else UNK for i in ids]
        out = []
        for i, tok in enumerate(parts):
            if tok == EOL:
                out.append('\n')
            elif i == 0 or out[-1] == '\n':
                out.append(tok)
            elif len(tok) == 1 and not tok.isalnum():
                # punctuation — attach directly, no leading space
                out.append(tok)
            else:
                out.append(' ' + tok)
        return ''.join(out)


def build_vocab(text, min_freq=3, vocab_path='word_vocab.txt'):
    """Count word frequencies and write a vocabulary file.

    Args:
        text:      raw training text
        min_freq:  minimum occurrence count to include a word (default 3)
        vocab_path: output file path

    Returns:
        WordTokenizer loaded from the new vocab file.
    """
    from collections import Counter

    # Tokenize to raw strings
    raw = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        raw.extend(_TOKEN_RE.findall(line))
        if i < len(lines) - 1 or line:
            raw.append(EOL)

    # Frequency count (exclude special tokens)
    counts = Counter(t for t in raw if t not in (UNK, EOL))
    kept = {t for t, c in counts.items() if c >= min_freq}

    # Vocabulary: specials first, then words sorted by descending frequency
    vocab = [UNK, EOL] + sorted(kept, key=lambda t: -counts[t])

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')

    total_words = sum(c for t, c in counts.items())
    covered = sum(c for t, c in counts.items() if t in kept)
    print(f"Vocabulary size : {len(vocab):,}  ({len(vocab) - 2:,} words + 2 special tokens)")
    print(f"Token coverage  : {covered / total_words * 100:.1f}% of word tokens in training set")
    print(f"Vocab written to: {vocab_path}")

    return WordTokenizer(vocab_path)
