import os
import pickle
import time
from pathlib import Path

import modal
from importlib import import_module

train_bpe = import_module("cs336_basics.2_0_bpe_tokenizer").train_bpe

DATA = Path("data")
INPUT = DATA / "owt_train.txt"
VOC, MEG, LOG, PROG = (
    "owt_vocab_32000.pkl",
    "owt_merges_32000.pkl",
    "train_bpe_expts_owt_log.txt",
    "train_bpe_expts_owt_progress.txt",
)
VMAX, ST = 32_000, ["<|endoftext|>"]
RWD, RDATA, ROUT = "/root/workspace", "/root/workspace/data", "/bpe_outputs"
VOL = modal.Volume.from_name("cs336-bpe-owt", create_if_missing=True)
IMG = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(DATA), remote_path=RDATA)
)
app = modal.App("train-bpe-expts-owt")


@app.function(image=IMG, volumes={ROUT: VOL}, timeout=24 * 60 * 60)
def train_owt_remote() -> str:
    os.chdir(RWD)
    inp = f"{RDATA}/{INPUT.name}"
    pp = Path(ROUT) / PROG
    pp.write_text("", encoding="utf-8")
    n = [0]

    def cb(msg: str):
        print(msg)
        with pp.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        n[0] += 1
        if n[0] >= 5:
            VOL.commit()
            n[0] = 0

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        inp,
        VMAX,
        ST,
        progress_interval_seconds=300,
        include_timestamps=True,
        progress_callback=cb,
    )
    el = time.perf_counter() - t0
    longest = max(vocab.values(), key=len)
    log = (
        f"input_path={INPUT}\nmax_vocab_size={VMAX}\nspecial_tokens={ST}\n"
        f"vocab_size={len(vocab)}\ntraining_time_seconds={el:.2f}\n"
        f"longest_token_length_bytes={len(longest)}\nlongest_token_utf8={longest.decode('utf-8', errors='replace')!r}\n"
    )
    for name, data, wb in ((VOC, vocab, True), (MEG, merges, True), (LOG, log, False)):
        p = Path(ROUT) / name
        if wb:
            pickle.dump(data, p.open("wb"))
        else:
            p.write_text(data, encoding="utf-8")
    VOL.commit()
    return log


@app.local_entrypoint()
def main() -> None:
    print(train_owt_remote.remote())
