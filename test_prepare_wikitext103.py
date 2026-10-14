"""Tests for prepare_wikitext103.py on a tiny synthetic zip (no download).

Run with ``.venv313/bin/python test_prepare_wikitext103.py``.
"""

import json
import os
import tempfile
import zipfile

import numpy as np
import tiktoken

import prepare_wikitext103 as P
from lut_torch._testing import run_all


def make_article(i, n_sections=3, reps=6):
    parts = [f" = Article {i} = \n", "\n"]
    for s in range(n_sections):
        parts.append(f" = = Section {s} = = \n\n")
        parts.append((f" The cat @-@ like thing , number {i} , sat on a mat . "
                      "It was 1,000 years old é . \n") * reps)
        parts.append("\n")
    return "".join(parts)


def build(tmp):
    train = "".join(make_article(i) for i in range(30))
    texts = {"wiki.train.raw": train, "wiki.valid.raw": make_article(100),
             "wiki.test.raw": make_article(200)}
    zp = os.path.join(tmp, "wt.zip")
    with zipfile.ZipFile(zp, "w") as z:
        for k, v in texts.items():
            z.writestr(f"wikitext-103-raw/{k}", v.encode("utf-8"))
    out = os.path.join(tmp, "out")
    P.main(["--zip-path", zp, "--out-dir", out, "--subset-words", "300", "900",
            "--wt2-val", os.path.join(tmp, "missing.txt")])
    return out, texts


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        out, texts = build(tmp)
        enc = tiktoken.get_encoding("gpt2")
        for txt, bn in [("train_raw.txt", "train_gpt2.bin"), ("val_raw.txt", "val_gpt2.bin"),
                        ("test_raw.txt", "test_gpt2.bin"), ("train_300w_raw.txt", "train_300w_gpt2.bin"),
                        ("train_900w_raw.txt", "train_900w_gpt2.bin")]:
            bp = os.path.join(out, bn)
            man = P.read_manifest(bp)
            text = open(os.path.join(out, txt), encoding="utf-8").read()
            ids = np.fromfile(bp, dtype=np.uint16)
            assert enc.decode(ids.tolist()) == text
            assert man["n_tokens"] == len(ids) and man["n_words"] == len(text.split())
            assert man["source_file"] == txt
        assert open(os.path.join(out, "train_raw.txt"), encoding="utf-8").read() == texts["wiki.train.raw"]
        for n in (300, 900):
            sub = open(os.path.join(out, f"train_{n}w_raw.txt"), encoding="utf-8").read()
            assert len(sub.split()) >= n
            assert texts["wiki.train.raw"].startswith(sub)
            rest = texts["wiki.train.raw"][len(sub):]
            assert P.ARTICLE_RE.match(rest.split("\n")[0]) and not rest.startswith(" = = ")
            # first boundary: dropping the last article would fall below n words
            last = sub.rfind(" = Article")
            assert len(sub[:last].split()) < n
        vman = json.load(open(os.path.join(out, "val_gpt2.bin.json")))
        assert "skipped" in vman["wt2_val_match"]
        assert os.path.exists(os.path.join(out, "README.txt"))


def test_idempotent_and_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        out, _ = build(tmp)
        bp = os.path.join(out, "val_gpt2.bin")
        before = os.path.getmtime(bp + ".json")
        zp = os.path.join(tmp, "wt.zip")
        P.main(["--zip-path", zp, "--out-dir", out, "--subset-words", "300", "900"])
        assert os.path.getmtime(bp + ".json") == before
        with open(bp, "ab") as f:
            f.write(b"\x00")
        try:
            P.read_manifest(bp)
        except ValueError as e:
            assert "size" in str(e)
        else:
            raise AssertionError("corrupted bin accepted")


def test_heading_regex():
    assert P.ARTICLE_RE.search(" = Title = \n")
    assert not P.ARTICLE_RE.search(" = = Sub = = \n")


def test_wt2_check():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "w2.txt")
        open(p, "w").write("a <unk> c d")
        res = P.wt2_val_check("a b c x", p)
        assert res["n_mismatches"] == 1 and res["first_mismatches"][0]["pos"] == 3


if __name__ == "__main__":
    raise SystemExit(run_all(globals()))
