import json
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import modal
import torch

from cs336_basics import train_lm as lm

M = import_module("cs336_basics.3_transformer_lm")
T = import_module("cs336_basics.2_4_tokenizer")
D = import_module("cs336_basics.5_decoding")

DEFAULT_CFG: dict[str, Any] = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1_344,
    "rope_theta": 10_000.0,
    "dtype": "float32",
}

# (dir, vocab.pkl, merges.pkl)
BPE: dict[str, tuple[str, str, str]] = {
    "tinystories": ("/bpe_tinystories_outputs", "tinystories_vocab_10000.pkl", "tinystories_merges_10000.pkl"),
    "owt": ("/bpe_owt_outputs", "owt_vocab_32000.pkl", "owt_merges_32000.pkl"),
}

IMG = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "einops", "regex")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(lm.DATA_DIR), remote_path=lm.REMOTE_DATA_DIR)
)
app = modal.App("generate-lm")
vol_ts = modal.Volume.from_name("cs336-bpe-tinystories", create_if_missing=False)
vol_owt = modal.Volume.from_name("cs336-bpe-owt", create_if_missing=False)


def _cfg(s: str) -> dict[str, Any]:
    c = dict(DEFAULT_CFG)
    if s.strip():
        c.update(json.loads(s))
    return c


def _norm_sd(sd: dict[str, Any]) -> dict[str, Any]:
    if not any(k.startswith("_orig_mod.") for k in sd):
        return sd
    return {k.removeprefix("_orig_mod.") if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}


def _model(cfg: dict[str, Any], device: str) -> torch.nn.Module:
    dt = torch.bfloat16 if str(cfg.get("dtype", "float32")).lower() == "bfloat16" else torch.float32
    return M.TransformerLM(
        vocab_size=int(cfg["vocab_size"]),
        context_length=int(cfg["context_length"]),
        d_model=int(cfg["d_model"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        d_ff=int(cfg["d_ff"]),
        rope_theta=float(cfg["rope_theta"]),
        device=device,
        dtype=dt,
    ).to(device)


@app.function(
    image=IMG,
    volumes={
        lm.REMOTE_OUTPUT_DIR: lm.output_volume,
        lm.REMOTE_TOKENIZER_OUTPUT_DIR: lm.tokenizer_output_volume,
        "/bpe_tinystories_outputs": vol_ts,
        "/bpe_owt_outputs": vol_owt,
    },
    timeout=7200,
    gpu="B200",
)
def run_generate_remote(
    checkpoint_path: str,
    prompt: str,
    config_json: str = "",
    tokenizer_source: str = "tinystories",
    special_token: str = "<|endoftext|>",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eos_token_id: int = -1,
    output_path: str = "",
) -> str:
    os.chdir(lm.REMOTE_WORKDIR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _cfg(config_json)
    ck = lm._to_remote_output_path(Path(checkpoint_path))
    key = tokenizer_source.strip().lower().removeprefix("cs336-bpe-")
    d, vf, mf = BPE[key]
    vp, mp = Path(d) / vf, Path(d) / mf
    tok = T.Tokenizer.from_files(str(vp), str(mp), special_tokens=[special_token])
    ids = tok.encode(prompt)
    eid = (
        tok.token_to_id[special_token.encode("utf-8")]
        if eos_token_id < 0
        else eos_token_id
    )
    m = _model(cfg, dev)
    cp = torch.load(ck, map_location="cpu")
    m.load_state_dict(_norm_sd(cp["model_state_dict"]))
    m.eval()
    out = D.decode(m, ids, eid, max_new_tokens, temperature, top_p)
    text = tok.decode(out)
    if output_path.strip():
        op = lm._to_remote_output_path(Path(output_path))
    else:
        op = lm._to_remote_output_path(Path(f"generations/generation_{int(time.time())}.txt"))
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(text, encoding="utf-8")
    lm.output_volume.commit()
    return f"Saved {op} prompt_tokens={len(ids)} total={len(out)}"


@app.local_entrypoint()
def main(
    checkpoint_path: str,
    prompt: str,
    config_json: str = "",
    tokenizer_source: str = "tinystories",
    special_token: str = "<|endoftext|>",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    eos_token_id: int = -1,
    output_path: str = "",
):
    print(
        run_generate_remote.remote(
            checkpoint_path=checkpoint_path,
            prompt=prompt,
            config_json=config_json,
            tokenizer_source=tokenizer_source,
            special_token=special_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            output_path=output_path,
        )
    )
