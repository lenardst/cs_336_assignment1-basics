import importlib
import os
from pathlib import Path

import modal
import numpy as np


APP_NAME = "check-modal-tokenizer-match"
REMOTE_WORKDIR = "/root/workspace"
REMOTE_DATA_DIR = f"{REMOTE_WORKDIR}/data"
REMOTE_TOKENIZER_OUTPUT_DIR = "/tokenizer_experiments_outputs"
OUTPUT_VOLUME_NAME = "cs336-tokenizer-experiments"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir("data", remote_path=REMOTE_DATA_DIR)
)
app = modal.App(APP_NAME)
tokenizer_output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=False)


DATASET_CONFIG = {
    "tiny_train": {
        "vocab": "tinystories_vocab_10000.pkl",
        "merges": "tinystories_merges_10000.pkl",
        "text": "TinyStoriesV2-GPT4-train.txt",
        "tokens": "tiny_train_uint16.npy",
    },
    "tiny_valid": {
        "vocab": "tinystories_vocab_10000.pkl",
        "merges": "tinystories_merges_10000.pkl",
        "text": "TinyStoriesV2-GPT4-valid.txt",
        "tokens": "tiny_valid_uint16.npy",
    },
    "owt_train": {
        "vocab": "owt_vocab_32000.pkl",
        "merges": "owt_merges_32000.pkl",
        "text": "owt_train.txt",
        "tokens": "owt_train_uint16.npy",
    },
    "owt_valid": {
        "vocab": "owt_vocab_32000.pkl",
        "merges": "owt_merges_32000.pkl",
        "text": "owt_valid.txt",
        "tokens": "owt_valid_uint16.npy",
    },
}


@app.function(
    image=image,
    volumes={REMOTE_TOKENIZER_OUTPUT_DIR: tokenizer_output_volume},
    timeout=60 * 60,
)
def check_match(
    dataset_key: str = "tiny_train",
    prefix_chars: int = 200_000,
    compare_tokens: int = 5_000,
) -> dict[str, int | float | str]:
    os.chdir(REMOTE_WORKDIR)
    if dataset_key not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset_key={dataset_key}. Expected one of: {sorted(DATASET_CONFIG)}")
    cfg = DATASET_CONFIG[dataset_key]

    tokenizer_module = importlib.import_module("cs336_basics.2_4_tokenizer")
    tokenizer_cls = getattr(tokenizer_module, "Tokenizer")
    tokenizer = tokenizer_cls.from_files(
        vocab_filepath=f"{REMOTE_DATA_DIR}/{cfg['vocab']}",
        merges_filepath=f"{REMOTE_DATA_DIR}/{cfg['merges']}",
        special_tokens=["<|endoftext|>"],
    )

    text_path = Path(f"{REMOTE_DATA_DIR}/{cfg['text']}")
    with text_path.open("r", encoding="utf-8") as f:
        text_prefix = f.read(prefix_chars)

    encoded = tokenizer.encode(text_prefix)
    arr = np.load(f"{REMOTE_TOKENIZER_OUTPUT_DIR}/{cfg['tokens']}", mmap_mode="r")

    k = min(compare_tokens, len(encoded), int(arr.shape[0]))
    encoded_prefix = np.array(encoded[:k], dtype=np.uint16)
    dataset_prefix = arr[:k]
    matches = int((encoded_prefix == dataset_prefix).sum())

    mismatch_index = -1
    if matches < k:
        mismatch_index = int(np.argmax(encoded_prefix != dataset_prefix))

    return {
        "dataset_key": dataset_key,
        "k": int(k),
        "matches": matches,
        "match_ratio": 0.0 if k == 0 else float(matches / k),
        "mismatch_index": mismatch_index,
    }


@app.local_entrypoint()
def main(
    dataset_key: str = "tiny_train",
    prefix_chars: int = 200_000,
    compare_tokens: int = 5_000,
) -> None:
    result = check_match.remote(
        dataset_key=dataset_key,
        prefix_chars=prefix_chars,
        compare_tokens=compare_tokens,
    )
    print(result)
