from __future__ import annotations

import os
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from importlib import import_module

import modal

train_bpe = import_module("cs336_basics.2_0_bpe_tokenizer").train_bpe

APP_NAME = "train-bpe-expts-owt"
DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "owt_train.txt"
VOCAB_PATH = DATA_DIR / "owt_vocab_32000.pkl"
MERGES_PATH = DATA_DIR / "owt_merges_32000.pkl"
LOG_PATH = DATA_DIR / "train_bpe_expts_owt_log.txt"
PROGRESS_LOG_PATH = DATA_DIR / "train_bpe_expts_owt_progress.txt"
MAX_VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_VOLUME_NAME = "cs336-bpe-owt"
PROGRESS_EVERY_MERGES: int | None = None
PROGRESS_EVERY_MINUTES = 5
PROGRESS_COMMIT_EVERY_UPDATES = 5

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


def _train(
    input_path: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], float]:
    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path,
        MAX_VOCAB_SIZE,
        SPECIAL_TOKENS,
        progress_every=PROGRESS_EVERY_MERGES,
        progress_interval_seconds=PROGRESS_EVERY_MINUTES * 60,
        include_timestamps=True,
        progress_callback=progress_callback,
    )
    elapsed_s = time.perf_counter() - start_time
    return vocab, merges, elapsed_s


@app.function(image=image, volumes={REMOTE_OUTPUT_DIR: output_volume}, timeout=24 * 60 * 60)
def train_owt_remote() -> str:
    os.chdir(REMOTE_WORKDIR)
    remote_input_path = f"{REMOTE_DATA_DIR}/{INPUT_PATH.name}"
    remote_progress_path = Path(REMOTE_OUTPUT_DIR) / PROGRESS_LOG_PATH.name
    remote_progress_path.write_text("", encoding="utf-8")
    updates_since_commit = 0

    def progress_callback(message: str) -> None:
        nonlocal updates_since_commit
        print(message)
        with remote_progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(f"{message}\n")
        updates_since_commit += 1
        if updates_since_commit >= PROGRESS_COMMIT_EVERY_UPDATES:
            output_volume.commit()
            updates_since_commit = 0

    vocab, merges, elapsed_s = _train(remote_input_path, progress_callback=progress_callback)
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
    try:
        deployed_train_fn = modal.Function.from_name(APP_NAME, "train_owt_remote")
    except Exception as exc:
        raise RuntimeError(
            "Could not find deployed Modal function 'train_owt_remote'. "
            f"Deploy first with: modal deploy {Path(__file__).as_posix()}"
        ) from exc

    function_call = deployed_train_fn.spawn()
    print("Submitted train_owt_remote job asynchronously.")
    print(f"FunctionCall ID: {function_call.object_id}")
    print(
        f"Remote artifacts will be saved to Modal Volume '{OUTPUT_VOLUME_NAME}': "
        f"{Path(REMOTE_OUTPUT_DIR) / VOCAB_PATH.name} and {Path(REMOTE_OUTPUT_DIR) / MERGES_PATH.name}"
    )
