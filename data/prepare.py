"""
Prepare FineWeb-Edu for nanoGPT-style training.

Streams HuggingFaceFW/fineweb-edu (sample-10BT), tokenizes with GPT-2 BPE,
and writes flat uint16 token files:
    train.bin, val.bin, test.bin  + meta.json (token and UTF-8 byte counts)

Byte counts are saved so bits-per-byte can be computed later:
    bpb = (loss_nats / ln 2) * (n_tokens / n_bytes)

Adapted from nanoGPT's data/openwebtext/prepare.py (MIT licence).
"""

import json
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset

# ---- config ----
OUT_DIR = "data/fineweb_edu"
TRAIN_TOKENS = (
    10_000_000  # 5_000_000_000  # ~10 GB on disk; enough for budgets up to ~3e17 FLOPs
)
VAL_DOCS = 5_000  # held-out docs for tuning
TEST_DOCS = 5_000  # held-out docs for final evaluation only
BATCH_DOCS = 1_000  # docs tokenized per batch
NUM_THREADS = 16  # tiktoken threads
# ----------------

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256, appended after every document


def encode_batch(texts):
    ids = enc.encode_ordinary_batch(texts, num_threads=NUM_THREADS)
    return [x + [EOT] for x in ids]


def write_split(path, docs_iter, max_docs=None, max_tokens=None):
    """Tokenize docs and append to a uint16 file. Returns (n_tokens, n_bytes, n_docs)."""
    n_tok = n_bytes = n_docs = 0
    batch = []
    with open(path, "wb") as f:

        def flush(batch):
            nonlocal n_tok, n_bytes
            for ids, text in zip(encode_batch(batch), batch):
                arr = np.asarray(
                    ids, dtype=np.uint16
                )  # GPT-2 vocab (50257) fits in uint16
                f.write(arr.tobytes())
                n_tok += len(arr)
                n_bytes += len(text.encode("utf-8"))

        for doc in docs_iter:
            batch.append(doc["text"])
            n_docs += 1
            if len(batch) == BATCH_DOCS:
                flush(batch)
                batch = []
                if max_tokens and n_tok >= max_tokens:
                    break
                if n_docs % 100_000 == 0:
                    print(f"  {path}: {n_docs:,} docs, {n_tok / 1e9:.2f}B tokens")
            if max_docs and n_docs >= max_docs:
                break
        if batch:
            flush(batch)
    return n_tok, n_bytes, n_docs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    ds = ds.shuffle(seed=1337, buffer_size=10_000)
    it = iter(ds)

    meta = {"tokenizer": "gpt2", "dtype": "uint16", "eot_token": EOT}

    # Val and test are taken first so they never overlap with train.
    for split, n in [("val", VAL_DOCS), ("test", TEST_DOCS)]:
        tok, byt, docs = write_split(
            os.path.join(OUT_DIR, f"{split}.bin"), it, max_docs=n
        )
        meta[split] = {
            "tokens": tok,
            "bytes": byt,
            "docs": docs,
            "tokens_per_byte": tok / byt,
        }
        print(f"{split}: {docs:,} docs, {tok:,} tokens, {byt:,} bytes")

    tok, byt, docs = write_split(
        os.path.join(OUT_DIR, "train.bin"), it, max_tokens=TRAIN_TOKENS
    )
    meta["train"] = {"tokens": tok, "bytes": byt, "docs": docs}
    print(f"train: {docs:,} docs, {tok:,} tokens")

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
    # The HF streaming iterator keeps background threads and HTTP connections
    # open when we stop reading early, which can block normal interpreter
    # shutdown. All files are closed and flushed above, so exit immediately.
    sys.stdout.flush()
    os._exit(0)
