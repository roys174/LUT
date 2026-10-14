"""Prepare WikiText-103 (RAW) tokenized with GPT-2 BPE for the LUT and nanoGPT trainers.

Usage::

    .venv313/bin/python prepare_wikitext103.py                    # download + everything
    .venv313/bin/python prepare_wikitext103.py --zip-path Z.zip   # use a local zip
    .venv313/bin/python prepare_wikitext103.py --force            # redo finished steps

Output (default ``nanoGPT_POS/data/wiki103/``): raw texts ``{train,val,test}_raw.txt``,
data-size-ladder prefixes ``train_{2M,8M,32M}_raw.txt``, their uint16 GPT-2 token files
``*_gpt2.bin``, a ``<name>.bin.json`` manifest next to every .bin, and a README.txt.

Why the pieces look the way they do:

* Subsets are *prefixes* cut right before a top-level article heading (`` = Title = ``), so
  a ladder rung never ends mid-article and each smaller rung is contained in the larger.
  Level-2 headings (`` = = Sub = = ``) are not article starts.
* Each text is encoded in ONE ``encode_ordinary`` call: chunking can change BPE merges at
  chunk boundaries, and the token stream must decode back to exactly the source text.
* The manifest records hashes of source text and token file, so a loader can prove the
  tokens belong to the text it thinks they do. ``read_manifest`` (below) is deliberately
  light (json/os/numpy) so ``lut_torch/data.py`` can import it.
* WikiText-2 and -103 are said to share a validation set; WT2 has rare words replaced by
  ``<unk>``, so the check compares word by word wherever WT2 is not ``<unk>``.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import zipfile

import numpy as np

MANIFEST_FORMAT = "lut-token-bin-v1"
DEFAULT_URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
DEFAULT_WT2_VAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "nanoGPT_POS", "data", "wiki2", "val_original_10_freq.txt")
ZIP_NAME = "wikitext-103-raw-v1.zip"
# Top-level heading only: " = Title = ". " = = Sub = = " has a second '=' and is excluded.
ARTICLE_RE = re.compile(r"^ = [^=].* = $", re.MULTILINE)


def read_manifest(bin_path):
    """Load and validate ``<bin_path>.json``; return the dict or raise ValueError."""
    mpath = bin_path + ".json"
    if not os.path.exists(mpath):
        raise ValueError(f"missing manifest {mpath}")
    with open(mpath) as f:
        man = json.load(f)
    if man.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{mpath}: format {man.get('format')!r} != {MANIFEST_FORMAT!r}")
    expected = int(man["n_tokens"]) * np.dtype(man["dtype"]).itemsize
    actual = os.path.getsize(bin_path)
    if actual != expected:
        raise ValueError(f"{bin_path}: size {actual} bytes != n_tokens*itemsize = {expected}"
                         " (truncated or corrupted token file)")
    return man


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    import requests
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)  # never leave a half-written file under the final name


def ensure_zip(args):
    if args.zip_path:
        return args.zip_path
    dest = os.path.join(args.out_dir, ZIP_NAME)
    if os.path.exists(dest) and zipfile.is_zipfile(dest) and not args.force:
        print(f"zip exists, skipping download: {dest}")
    else:
        print(f"downloading {args.url} -> {dest}")
        download(args.url, dest)
    print(f"zip size: {os.path.getsize(dest) / 1e6:.1f} MB")
    return dest


def extract_raw(zip_path, out_dir, force):
    wanted = {"wiki.train.raw": "train_raw.txt", "wiki.valid.raw": "val_raw.txt",
              "wiki.test.raw": "test_raw.txt"}
    with zipfile.ZipFile(zip_path) as z:
        by_base = {os.path.basename(n): n for n in z.namelist()}
        for base, out in wanted.items():
            if base not in by_base:
                raise SystemExit(f"{base} not found in {zip_path}")
            dest = os.path.join(out_dir, out)
            if os.path.exists(dest) and not force:
                print(f"exists, skipping: {out}")
                continue
            with z.open(by_base[base]) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)  # unchanged bytes
            print(f"extracted {out} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return {v: os.path.join(out_dir, v) for v in wanted.values()}


def article_prefix_end(text, n_words):
    """Char offset of the first article heading at or after `n_words` words (or len(text)).

    Words are counted segment by segment between headings; every cut is at a line start,
    so no word is split and the sum equals len(text[:pos].split()).
    """
    words = 0
    prev = 0
    for m in ARTICLE_RE.finditer(text):
        words += len(text[prev:m.start()].split())
        prev = m.start()
        if words >= n_words:
            return m.start()
    return len(text)  # not enough text for a boundary: caller warns


def subset_label(n):
    return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n}w"


def write_subsets(train_path, subset_words, out_dir, force):
    """Return [(txt_name, bin_name)] for the ladder, writing missing prefix files."""
    pairs = []
    text = None
    for n in subset_words:
        lab = subset_label(n)
        txt_name, bin_name = f"train_{lab}_raw.txt", f"train_{lab}_gpt2.bin"
        pairs.append((txt_name, bin_name))
        dest = os.path.join(out_dir, txt_name)
        if os.path.exists(dest) and not force:
            print(f"exists, skipping: {txt_name}")
            continue
        if text is None:
            with open(train_path, encoding="utf-8") as f:
                text = f.read()
        end = article_prefix_end(text, n)
        if end == len(text):
            print(f"WARNING: {txt_name}: no article boundary after {n} words; using all of train")
        with open(dest, "w", encoding="utf-8", newline="") as f:
            f.write(text[:end])
        print(f"wrote {txt_name}: {len(text[:end].split()):,} words")
    return pairs


def wt2_val_check(val_text, wt2_path, max_show=5):
    if not wt2_path or not os.path.exists(wt2_path):
        res = {"skipped": f"WT2 val file not found: {wt2_path}"}
        print("WT2 val check: skipped (file not found)")
        return res
    with open(wt2_path, encoding="utf-8") as f:
        w2 = f.read().split()
    w103 = val_text.split()
    mism = [(i, a, b) for i, (a, b) in enumerate(zip(w2, w103)) if a != "<unk>" and a != b]
    res = {"wt2_words": len(w2), "wt103_words": len(w103),
           "word_counts_equal": len(w2) == len(w103),
           "wt2_unk": sum(1 for w in w2 if w == "<unk>"),
           "n_mismatches": len(mism),
           "first_mismatches": [{"pos": i, "wt2": a, "wt103": b} for i, a, b in mism[:max_show]]}
    print(f"WT2 val check: words wt2={len(w2):,} wt103={len(w103):,} "
          f"equal={res['word_counts_equal']}; non-<unk> mismatches={len(mism)}")
    for i, a, b in mism[:max_show]:
        print(f"  mismatch at word {i}: wt2={a!r} wt103={b!r}")
    return res


def tokenize_file(out_dir, txt_name, bin_name, zip_url, wt2_val, force):
    bin_path = os.path.join(out_dir, bin_name)
    if os.path.exists(bin_path) and os.path.exists(bin_path + ".json") and not force:
        try:
            read_manifest(bin_path)
            print(f"exists, skipping: {bin_name}")
            return
        except ValueError as e:
            print(f"redoing {bin_name}: {e}")
    import tiktoken
    txt_path = os.path.join(out_dir, txt_name)
    with open(txt_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)  # whole text in one call, see module docstring
    assert max(ids) < 65536, "token id does not fit uint16"
    arr = np.asarray(ids, dtype=np.uint16)
    del ids
    arr.tofile(bin_path)
    n_words = len(text.split())
    man = {"format": MANIFEST_FORMAT, "tokenizer": "gpt2", "dtype": "uint16",
           "n_tokens": int(arr.size), "n_words": n_words,
           "tokens_per_word": float(arr.size / max(n_words, 1)),
           "source_file": txt_name, "source_sha256": hashlib.sha256(raw).hexdigest(),
           "tokens_sha256": sha256_file(bin_path), "tiktoken_version": tiktoken.__version__,
           "zip_url": zip_url,
           "created": datetime.datetime.now().isoformat(timespec="seconds")}
    if txt_name == "val_raw.txt":
        man["wt2_val_match"] = wt2_val_check(text, wt2_val)
    with open(bin_path + ".json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"wrote {bin_name}: {arr.size:,} tokens, {n_words:,} words")
    del text, raw, arr


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default=os.path.join("nanoGPT_POS", "data", "wiki103"))
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--zip-path", default=None, help="use this zip, skip download")
    p.add_argument("--subset-words", type=int, nargs="+",
                   default=[2_000_000, 8_000_000, 32_000_000])
    p.add_argument("--wt2-val", default=DEFAULT_WT2_VAL,
                   help="WikiText-2 val file for the shared-validation-set check")
    p.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cmd = " ".join([sys.executable, "prepare_wikitext103.py"] +
                   (sys.argv[1:] if argv is None else list(argv)))
    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = ensure_zip(args)
    extract_raw(zip_path, args.out_dir, args.force)
    subsets = write_subsets(os.path.join(args.out_dir, "train_raw.txt"),
                            args.subset_words, args.out_dir, args.force)
    zip_url = None if args.zip_path else args.url
    files = [("train_raw.txt", "train_gpt2.bin")] + subsets + \
            [("val_raw.txt", "val_gpt2.bin"), ("test_raw.txt", "test_gpt2.bin")]
    for txt_name, bin_name in files:
        tokenize_file(args.out_dir, txt_name, bin_name, zip_url, args.wt2_val, args.force)

    rows = []
    for _, bin_name in files:
        man = read_manifest(os.path.join(args.out_dir, bin_name))
        rows.append((bin_name, man["n_tokens"], man["n_words"], man["tokens_per_word"]))
    table = [f"{'file':<22}{'tokens':>14}{'words':>14}{'tok/word':>10}"]
    table += [f"{n:<22}{t:>14,}{w:>14,}{r:>10.4f}" for n, t, w, r in rows]
    table = "\n".join(table)
    print(table)
    with open(os.path.join(args.out_dir, "README.txt"), "w") as f:
        f.write("WikiText-103 (raw), GPT-2 BPE (tiktoken gpt2), uint16 token files.\n"
                "Each X.bin has a manifest X.bin.json (format lut-token-bin-v1; see\n"
                "read_manifest in prepare_wikitext103.py). train_{2M,8M,32M}_* are article-\n"
                "aligned prefixes of train_raw.txt (>= N whitespace words).\n\n"
                f"Command: {cmd}\nCreated: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
                f"{table}\n")


if __name__ == "__main__":
    main()
