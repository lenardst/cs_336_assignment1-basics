import os
from importlib import import_module

import modal
import numpy as np

Tok = import_module("cs336_basics.2_4_tokenizer").Tokenizer
WD, DATA, VOL = "/root/workspace", "/root/workspace/data", "/tokenizer_experiments_outputs"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir("data", remote_path=DATA)
)
app = modal.App("check-modal-decoded-prefix")
vol = modal.Volume.from_name("cs336-tokenizer-experiments", create_if_missing=False)


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def run(n: int = 300) -> dict:
    os.chdir(WD)
    tok = Tok.from_files(
        f"{DATA}/tinystories_vocab_10000.pkl",
        f"{DATA}/tinystories_merges_10000.pkl",
        special_tokens=["<|endoftext|>"],
    )
    ids = np.load(f"{VOL}/tiny_train_uint16.npy", mmap_mode="r")[:n].astype(int).tolist()
    dec = tok.decode(ids)
    raw = open(f"{DATA}/TinyStoriesV2-GPT4-train.txt", encoding="utf-8").read(max(len(dec), 1200))
    L = min(len(dec), len(raw))
    mm = next((i for i in range(L) if dec[i] != raw[i]), -1)
    return {"n": n, "first_mismatch": mm, "dec_preview": dec[:400], "raw_preview": raw[:400]}


@app.local_entrypoint()
def main(n: int = 300):
    print(run.remote(n=n))
