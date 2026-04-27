import argparse
import os
import pickle
import resource
import time
from importlib import import_module
from pathlib import Path

import modal

train_bpe = import_module("cs336_basics.2_0_bpe_tokenizer").train_bpe

DATA = Path("data")
INPUT = DATA / "TinyStoriesV2-GPT4-train.txt"
ST = ["<|endoftext|>"]
RWD, RDATA, ROUT = "/root/workspace", "/root/workspace/data", "/bpe_outputs"
VOL = modal.Volume.from_name("cs336-bpe-tinystories", create_if_missing=True)
IMG = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(DATA), remote_path=RDATA)
)
app = modal.App("train-bpe-tinystories")


def names(n: int):
    return (
        f"tinystories_vocab_{n}.pkl",
        f"tinystories_merges_{n}.pkl",
        f"train_bpe_tinystories_{n}_log.txt",
        f"train_bpe_tinystories_{n}_progress.txt",
    )


@app.function(image=IMG, volumes={ROUT: VOL}, timeout=86400)
def train_tinystories_remote(n: int) -> str:
    os.chdir(RWD)
    vn, mn, ln, pn = names(n)
    prog = n >= 16_000
    cb = None
    if prog:
        pp = Path(ROUT) / pn
        pp.write_text("", encoding="utf-8")
        cnt = [0]

        def cb(msg: str):
            print(msg)
            with pp.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            cnt[0] += 1
            if cnt[0] >= 5:
                VOL.commit()
                cnt[0] = 0

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        f"{RDATA}/{INPUT.name}",
        n,
        ST,
        progress_interval_seconds=300 if prog else None,
        include_timestamps=prog,
        progress_callback=cb,
    )
    el = time.perf_counter() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    longest = max(vocab.values(), key=len).decode("utf-8", errors="replace")
    log = (
        f"input_path={INPUT}\nmax_vocab_size={n}\nspecial_tokens={ST}\n"
        f"vocab_size={len(vocab)}\ntraining_time_seconds={el:.2f}\n"
        f"peak_memory_mb={peak_kb/1024:.1f}\nlongest_token={longest!r}\n"
    )
    for fn, data, binary in ((vn, vocab, True), (mn, merges, True), (ln, log, False)):
        p = Path(ROUT) / fn
        if binary:
            pickle.dump(data, p.open("wb"))
        else:
            p.write_text(data, encoding="utf-8")
    VOL.commit()
    return log


@app.local_entrypoint()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=10_000)
    n = p.parse_known_args()[0].vocab_size
    log = train_tinystories_remote.remote(n)
    vn, mn, ln, _ = names(n)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / ln).write_text(log, encoding="utf-8")
    print(f"Wrote {DATA / ln}; volume: {ROUT}/{vn}, {ROUT}/{mn}")
