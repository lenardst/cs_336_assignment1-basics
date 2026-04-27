import os
from importlib import import_module
from pathlib import Path

import modal
import numpy as np

Tok = import_module("cs336_basics.2_4_tokenizer").Tokenizer
WD, DATA, VOL = "/root/workspace", "/root/workspace/data", "/tokenizer_experiments_outputs"
CFG = {
    "tiny_train": ("tinystories_vocab_10000.pkl", "tinystories_merges_10000.pkl", "TinyStoriesV2-GPT4-train.txt", "tiny_train_uint16.npy"),
    "tiny_valid": ("tinystories_vocab_10000.pkl", "tinystories_merges_10000.pkl", "TinyStoriesV2-GPT4-valid.txt", "tiny_valid_uint16.npy"),
    "owt_train": ("owt_vocab_32000.pkl", "owt_merges_32000.pkl", "owt_train.txt", "owt_train_uint16.npy"),
    "owt_valid": ("owt_vocab_32000.pkl", "owt_merges_32000.pkl", "owt_valid.txt", "owt_valid_uint16.npy"),
}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir("data", remote_path=DATA)
)
app = modal.App("check-modal-tokenizer-match")
vol = modal.Volume.from_name("cs336-tokenizer-experiments", create_if_missing=False)


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def run(key: str = "tiny_train", chars: int = 200_000, k: int = 5000) -> dict:
    os.chdir(WD)
    voc, meg, txt, npy = CFG[key]
    tok = Tok.from_files(f"{DATA}/{voc}", f"{DATA}/{meg}", special_tokens=["<|endoftext|>"])
    text = Path(f"{DATA}/{txt}").read_text(encoding="utf-8")[:chars]
    enc = np.array(tok.encode(text)[:k], dtype=np.uint16)
    arr = np.load(f"{VOL}/{npy}", mmap_mode="r")[:k]
    kk = min(len(enc), len(arr))
    enc, arr = enc[:kk], arr[:kk]
    ok = enc == arr
    return {
        "key": key,
        "k": kk,
        "matches": int(ok.sum()),
        "mismatch": int(np.argmax(~ok)) if kk and not ok.all() else -1,
    }


@app.local_entrypoint()
def main(key: str = "tiny_train"):
    print(run.remote(key=key))
