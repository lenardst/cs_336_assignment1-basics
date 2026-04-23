from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from importlib import import_module

import modal

train_bpe = import_module("cs336_basics.2_0_bpe_tokenizer").train_bpe

APP_NAME = "train-bpe-tinystories"
DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VOCAB_PATH = DATA_DIR / "tinystories_vocab_10000.pkl"
MERGES_PATH = DATA_DIR / "tinystories_merges_10000.pkl"
LOG_PATH = DATA_DIR / "train_bpe_tinystories_log.txt"
MAX_VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_VOLUME_NAME = "cs336-bpe-tinystories"

REMOTE_WORKDIR = "/root/workspace"
REMOTE_DATA_DIR = f"{REMOTE_WORKDIR}/data"
REMOTE_OUTPUT_DIR = "/bpe_outputs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(DATA_DIR), remote_path=REMOTE_DATA_DIR)
)
app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _format_log(elapsed_s: float, vocab_size: int, longest_token: bytes) -> str:
    longest_token_text = longest_token.decode("utf-8", errors="replace")
    return (
        f"input_path={INPUT_PATH}\n"
        f"max_vocab_size={MAX_VOCAB_SIZE}\n"
        f"special_tokens={SPECIAL_TOKENS}\n"
        f"vocab_size={vocab_size}\n"
        f"training_time_seconds={elapsed_s:.2f}\n"
        f"longest_token_length_bytes={len(longest_token)}\n"
        f"longest_token_utf8={longest_token_text!r}\n"
    )


def _train(input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], float]:
    start_time = time.perf_counter()
    vocab, merges = train_bpe(input_path, MAX_VOCAB_SIZE, SPECIAL_TOKENS)
    elapsed_s = time.perf_counter() - start_time
    return vocab, merges, elapsed_s


@app.function(image=image, volumes={REMOTE_OUTPUT_DIR: output_volume}, timeout=24 * 60 * 60)
def train_tinystories_remote() -> str:
    os.chdir(REMOTE_WORKDIR)
    remote_input_path = f"{REMOTE_DATA_DIR}/{INPUT_PATH.name}"
    vocab, merges, elapsed_s = _train(remote_input_path)
    longest_token = max(vocab.values(), key=len)
    log_text = _format_log(elapsed_s, len(vocab), longest_token)

    remote_vocab_path = Path(REMOTE_OUTPUT_DIR) / VOCAB_PATH.name
    remote_merges_path = Path(REMOTE_OUTPUT_DIR) / MERGES_PATH.name
    remote_log_path = Path(REMOTE_OUTPUT_DIR) / LOG_PATH.name
    with remote_vocab_path.open("wb") as vocab_file:
        pickle.dump(vocab, vocab_file)
    with remote_merges_path.open("wb") as merges_file:
        pickle.dump(merges, merges_file)
    with remote_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(log_text)
    output_volume.commit()
    return log_text


@app.local_entrypoint()
def main() -> None:
    log_text = train_tinystories_remote.remote()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        log_file.write(log_text)

    print(f"Wrote: {LOG_PATH}")
    print(
        f"Remote artifacts saved to Modal Volume '{OUTPUT_VOLUME_NAME}': "
        f"{Path(REMOTE_OUTPUT_DIR) / VOCAB_PATH.name} and {Path(REMOTE_OUTPUT_DIR) / MERGES_PATH.name}"
    )
