import importlib
import os

import modal
import numpy as np


APP_NAME = "check-modal-decoded-prefix"
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


def _char_match_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


@app.function(
    image=image,
    volumes={REMOTE_TOKENIZER_OUTPUT_DIR: tokenizer_output_volume},
    timeout=60 * 60,
)
def check_decoded_prefix(token_count: int = 300, preview_chars: int = 1200) -> dict[str, object]:
    os.chdir(REMOTE_WORKDIR)
    tokenizer_module = importlib.import_module("cs336_basics.2_4_tokenizer")
    tokenizer_cls = getattr(tokenizer_module, "Tokenizer")
    tokenizer = tokenizer_cls.from_files(
        vocab_filepath=f"{REMOTE_DATA_DIR}/tinystories_vocab_10000.pkl",
        merges_filepath=f"{REMOTE_DATA_DIR}/tinystories_merges_10000.pkl",
        special_tokens=["<|endoftext|>"],
    )

    arr = np.load(f"{REMOTE_TOKENIZER_OUTPUT_DIR}/tiny_train_uint16.npy", mmap_mode="r")
    ids = arr[:token_count].astype(int).tolist()
    decoded = tokenizer.decode(ids)

    with open(f"{REMOTE_DATA_DIR}/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        raw_text = f.read(max(len(decoded), preview_chars))

    compare_len = min(len(decoded), len(raw_text))
    decoded_cmp = decoded[:compare_len]
    raw_cmp = raw_text[:compare_len]

    first_mismatch = -1
    for i in range(compare_len):
        if decoded_cmp[i] != raw_cmp[i]:
            first_mismatch = i
            break

    return {
        "token_count": token_count,
        "decoded_length": len(decoded),
        "compare_len": compare_len,
        "char_match_ratio": _char_match_ratio(decoded_cmp, raw_cmp),
        "first_mismatch_index": first_mismatch,
        "decoded_preview": decoded[:preview_chars],
        "raw_preview": raw_text[:preview_chars],
    }


@app.local_entrypoint()
def main(token_count: int = 300, preview_chars: int = 1200) -> None:
    result = check_decoded_prefix.remote(token_count=token_count, preview_chars=preview_chars)
    print(result)
