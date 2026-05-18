"""Download and tokenize WikiText-2 for nanoGPT.

Run from the LUT repo root:
    python nanoGPT_configs/prepare_wiki2.py

Writes train.bin and val.bin into nanoGPT_POS/data/wiki2/.
"""

import os
import sys
import numpy as np
# import tiktoken
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from word_tokenizer import WordTokenizer

# OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'nanoGPT', 'data', 'wiki2')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'nanoGPT_POS', 'data', 'wiki2')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vocab_files', '20_freq_wiki2_word_tok.txt')

URLS = {
    'train': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt',
    'val':   'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt',
}

os.makedirs(OUT_DIR, exist_ok=True)

POS_task = True
if POS_task:
    enc = WordTokenizer(VOCAB_PATH)
else:
    enc = tiktoken.get_encoding('gpt2')

for split, url in URLS.items():
    txt_path = os.path.join(OUT_DIR, f'{split}.txt')
    if not os.path.exists(txt_path):
        print(f'Downloading {split}...')
        r = requests.get(url)
        r.raise_for_status()
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(r.text)

    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if POS_task:
        ids = enc.encode(text)
    else:
        ids = enc.encode_ordinary(text)
    arr = np.array(ids, dtype=np.uint16)
    out_path = os.path.join(OUT_DIR, f'{split}.bin')
    arr.tofile(out_path)
    print(f'{split}: {len(ids):,} tokens -> {out_path}')
