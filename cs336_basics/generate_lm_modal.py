import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal
import torch

from cs336_basics import train_lm as lm

transformer_module = importlib.import_module("cs336_basics.3_transformer_lm")
tokenizer_module = importlib.import_module("cs336_basics.2_4_tokenizer")
decoding_module = importlib.import_module("cs336_basics.5_decoding")

TransformerLM = getattr(transformer_module, "TransformerLM")
Tokenizer = getattr(tokenizer_module, "Tokenizer")
decode = getattr(decoding_module, "decode")

APP_NAME = "generate-lm"
SPECIAL_TOKEN = "<|endoftext|>"

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1_344,
    "rope_theta": 10_000.0,
    "dtype": "float32",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "einops", "regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(lm.DATA_DIR), remote_path=lm.REMOTE_DATA_DIR)
)
app = modal.App(APP_NAME)


def _load_model_config(config_json: str) -> dict[str, Any]:
    if not config_json.strip():
        return dict(DEFAULT_MODEL_CONFIG)
    loaded = json.loads(config_json)
    if not isinstance(loaded, dict):
        raise ValueError("config_json must decode to a JSON object.")
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update(loaded)
    return config


def _build_model(config: dict[str, Any], device: str) -> torch.nn.Module:
    dtype_str = str(config.get("dtype", "float32")).lower()
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=dtype,
    )
    return model.to(device)


def _resolve_remote_input_path(path_str: str) -> Path:
    return lm._to_remote_input_path(Path(path_str))


def _resolve_remote_output_path(path_str: str) -> Path:
    return lm._to_remote_output_path(Path(path_str))


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            stripped[key[len("_orig_mod."):]] = value
        else:
            stripped[key] = value
    return stripped


@app.function(
    image=image,
    volumes={
        lm.REMOTE_OUTPUT_DIR: lm.output_volume,
        lm.REMOTE_TOKENIZER_OUTPUT_DIR: lm.tokenizer_output_volume,
    },
    timeout=2 * 60 * 60,
    gpu="B200",
)
def run_generate_remote(
    checkpoint_path: str,
    prompt: str,
    config_json: str = "",
    vocab_path: str = "data/tinystories_vocab_10000.pkl",
    merges_path: str = "data/tinystories_merges_10000.pkl",
    special_token: str = SPECIAL_TOKEN,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eos_token_id: int = -1,
    output_path: str = "",
) -> str:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not (0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if not prompt:
        raise ValueError("prompt must be non-empty.")

    os.chdir(lm.REMOTE_WORKDIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _load_model_config(config_json)

    checkpoint_remote = _resolve_remote_output_path(checkpoint_path)
    vocab_remote = _resolve_remote_input_path(vocab_path)
    merges_remote = _resolve_remote_input_path(merges_path)

    if not checkpoint_remote.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_remote}")
    if not vocab_remote.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_remote}")
    if not merges_remote.exists():
        raise FileNotFoundError(f"Merges file not found: {merges_remote}")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_remote),
        merges_filepath=str(merges_remote),
        special_tokens=[special_token],
    )
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("Prompt encoded to zero tokens; choose a different prompt.")

    if eos_token_id < 0:
        eos_bytes = special_token.encode("utf-8")
        eos_token_id = tokenizer.token_to_id[eos_bytes]

    model = _build_model(config, device=device)
    checkpoint = torch.load(checkpoint_remote, map_location="cpu")
    model_state = _normalize_state_dict_keys(checkpoint["model_state_dict"])
    model.load_state_dict(model_state)
    model.eval()
    print(f"[checkpoint] loaded iteration={checkpoint['iteration']}")

    generated_ids = decode(
        model=model,
        prompt_tokens=prompt_tokens,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generated_text = tokenizer.decode(generated_ids)

    if output_path.strip():
        output_remote = _resolve_remote_output_path(output_path)
    else:
        timestamp = int(time.time())
        output_remote = _resolve_remote_output_path(f"generations/generation_{timestamp}.txt")
    output_remote.parent.mkdir(parents=True, exist_ok=True)
    output_remote.write_text(generated_text, encoding="utf-8")
    lm.output_volume.commit()

    return (
        f"Saved generation to {output_remote}. "
        f"prompt_tokens={len(prompt_tokens)} total_tokens={len(generated_ids)}"
    )


@app.local_entrypoint()
def main(
    checkpoint_path: str,
    prompt: str,
    config_json: str = "",
    vocab_path: str = "data/tinystories_vocab_10000.pkl",
    merges_path: str = "data/tinystories_merges_10000.pkl",
    special_token: str = SPECIAL_TOKEN,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eos_token_id: int = -1,
    output_path: str = "",
) -> None:
    result = run_generate_remote.remote(
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        config_json=config_json,
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_token=special_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        output_path=output_path,
    )
    print(result)
