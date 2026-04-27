import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch

Tm = importlib.import_module("cs336_basics.3_transformer_lm")
Ab = importlib.import_module("cs336_basics.7_ablations")
Tr = importlib.import_module("cs336_basics.4_training")

TransformerLM = Tm.TransformerLM
cross_entropy, get_batch = Tr.cross_entropy, Tr.get_batch
AdamW, cosine_lr_wup = Tr.AdamW, Tr.cosine_lr_wup
gradient_clipping, save_checkpoint, load_checkpoint = (
    Tr.gradient_clipping,
    Tr.save_checkpoint,
    Tr.load_checkpoint,
)

MODEL_CLASS_BY_VARIANT = {
    "baseline": TransformerLM,
    "layer_norm_ablation": Ab.layer_norm_ablation_transformer_model,
    "pre_norm_ablation": Ab.pre_norm_ablation_transformer_model,
    "no_pos_emb": Ab.no_pos_emb_transformer_model,
    "swiglu_ablation": Ab.swiglu_ablation_transformer_model,
}

EXPERIMENT_TO_MODEL_VARIANT = {
    "baseline": "baseline",
    "layer_norm_ablation": "layer_norm_ablation",
    "pre_norm_ablation": "pre_norm_ablation",
    "no_pos_emb": "no_pos_emb",
    "swiglu_ablation": "swiglu_ablation",
}

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DEFAULT_MODAL_DEVICE = "cuda"
DEFAULT_MAX_ITERS = 40_000 if DEFAULT_DEVICE == "cuda" else 5_000

# Assignment baseline defaults.
DEFAULTS = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1_344,
    "rope_theta": 10_000.0,
    "batch_size": 64,
    "max_iters": DEFAULT_MAX_ITERS,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_iters": max(1, DEFAULT_MAX_ITERS // 10),
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "eval_interval": 200,
    "eval_batches": 20,
    "checkpoint_interval": 5000,
    "train_log_interval": 100,
}

APP_NAME = "train-lm"
DATA_DIR = Path("data")
OUTPUT_VOLUME_NAME = "cs336-lm-training"
TOKENIZER_OUTPUT_VOLUME_NAME = "cs336-tokenizer-experiments"
REMOTE_WORKDIR = "/root/workspace"
REMOTE_DATA_DIR = f"{REMOTE_WORKDIR}/data"
REMOTE_OUTPUT_DIR = "/lm_training_outputs"
REMOTE_TOKENIZER_OUTPUT_DIR = "/tokenizer_experiments_outputs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "einops")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(DATA_DIR), remote_path=REMOTE_DATA_DIR)
)
app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
tokenizer_output_volume = modal.Volume.from_name(TOKENIZER_OUTPUT_VOLUME_NAME, create_if_missing=False)


TRAIN_CONFIG_DEFAULTS: dict[str, Any] = {
    "experiment": "baseline",
    "model_variant": "baseline",
    "run_name": "",
    "train_path": f"{REMOTE_TOKENIZER_OUTPUT_DIR}/tiny_train_uint16.npy",
    "val_path": f"{REMOTE_TOKENIZER_OUTPUT_DIR}/tiny_valid_uint16.npy",
    "device": DEFAULT_MODAL_DEVICE,
    "dtype": "float32",
    "vocab_size": DEFAULTS["vocab_size"],
    "context_length": DEFAULTS["context_length"],
    "d_model": DEFAULTS["d_model"],
    "num_layers": DEFAULTS["num_layers"],
    "num_heads": DEFAULTS["num_heads"],
    "d_ff": DEFAULTS["d_ff"],
    "rope_theta": DEFAULTS["rope_theta"],
    "batch_size": DEFAULTS["batch_size"],
    "max_iters": DEFAULTS["max_iters"],
    "learning_rate": DEFAULTS["learning_rate"],
    "min_learning_rate": DEFAULTS["min_lr"],
    "warmup_iters": DEFAULTS["warmup_iters"],
    "beta1": DEFAULTS["beta1"],
    "beta2": DEFAULTS["beta2"],
    "epsilon": DEFAULTS["epsilon"],
    "weight_decay": DEFAULTS["weight_decay"],
    "max_grad_norm": DEFAULTS["max_grad_norm"],
    "eval_interval": DEFAULTS["eval_interval"],
    "eval_batches": DEFAULTS["eval_batches"],
    "train_log_interval": DEFAULTS["train_log_interval"],
    "checkpoint_interval": DEFAULTS["checkpoint_interval"],
    "checkpoint_path": "checkpoints/lm.pt",
    "log_path": "logs/train_log.jsonl",
    "enable_wandb": False,
    "wandb_project": "cs336-assignment1",
    "wandb_entity": "",
    "wandb_mode": "online",
    "resume": False,
    "compile": "off",
}


def _resolve_train_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(TRAIN_CONFIG_DEFAULTS)
    resolved_overrides = overrides or {}
    if resolved_overrides:
        config.update(resolved_overrides)
    if "model_variant" not in resolved_overrides:
        config["model_variant"] = EXPERIMENT_TO_MODEL_VARIANT.get(
            str(config.get("experiment", "baseline")),
            "baseline",
        )
    if "warmup_iters" not in resolved_overrides:
        config["warmup_iters"] = max(1, int(config["max_iters"]) // 10)
    if not config["run_name"]:
        config["run_name"] = f"run_{int(time.time())}"
    return config


def load_token_array(path: Path) -> np.ndarray | np.memmap:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r+")
    return np.memmap(path, dtype=np.uint16, mode="r+")


def load_train_val_arrays(train_path: Path, val_path: Path) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap]:
    return load_token_array(train_path), load_token_array(val_path)


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model_kwargs: dict[str, object] = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "device": args.device,
        "dtype": dtype,
    }

    model_variant = str(getattr(args, "model_variant", "baseline")).lower()
    model = MODEL_CLASS_BY_VARIANT[model_variant](**model_kwargs)
    if args.compile == "default":
        model = torch.compile(model)
    elif args.compile == "aot_eager":
        model = torch.compile(model, backend="aot_eager")
    return model


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data: np.ndarray | np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x.long())
        losses.append(cross_entropy(logits, y.long()).item())
    model.train()
    return float(np.mean(losses))


def _append_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _to_remote_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == DATA_DIR.name:
        return Path(REMOTE_DATA_DIR).joinpath(*path.parts[1:])
    return Path(REMOTE_WORKDIR) / path


def _to_remote_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(REMOTE_OUTPUT_DIR) / path


def _namespace_from_config(config: dict[str, Any], remote: bool) -> argparse.Namespace:
    resolved = _resolve_train_config(config)
    device = _normalize_remote_device(str(resolved["device"])) if remote else str(resolved["device"])
    train_path = Path(str(resolved["train_path"]))
    val_path = Path(str(resolved["val_path"]))
    checkpoint_path = Path(str(resolved["checkpoint_path"]))
    log_path = Path(str(resolved["log_path"]))
    if remote:
        train_path = _to_remote_input_path(train_path)
        val_path = _to_remote_input_path(val_path)
        checkpoint_path = _to_remote_output_path(checkpoint_path)
        log_path = _to_remote_output_path(log_path)
    resolved["device"] = device
    resolved["train_path"] = train_path
    resolved["val_path"] = val_path
    resolved["checkpoint_path"] = checkpoint_path
    resolved["log_path"] = log_path
    return argparse.Namespace(**resolved)


def train(args: argparse.Namespace) -> None:
    print(args.run_name, args.device, args.dtype, args.max_iters)
    if args.device.startswith("cuda") and args.dtype == "float32":
        torch.set_float32_matmul_precision("high")
    train_data, val_data = load_train_val_arrays(args.train_path, args.val_path)
    model = _build_model(args)
    print("params", sum(p.numel() for p in model.parameters()))
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        lamb=args.weight_decay,
        epsilon=args.epsilon,
    )

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_iter = 0
    if args.resume and args.checkpoint_path.exists():
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)

    t0 = time.perf_counter()
    tokens_per_step = args.batch_size * args.context_length
    final_step_count = start_iter
    model.train()
    wandb_run = None
    if bool(getattr(args, "enable_wandb", False)):
        import wandb

        wandb_kwargs: dict[str, Any] = {
            "project": str(getattr(args, "wandb_project", "cs336-assignment1")),
            "name": args.run_name,
            "config": vars(args),
            "mode": str(getattr(args, "wandb_mode", "online")),
            "reinit": True,
        }
        wandb_entity = str(getattr(args, "wandb_entity", "")).strip()
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        wandb_run = wandb.init(**wandb_kwargs)
    for i in range(start_iter, args.max_iters):
        step_count = i + 1
        total_tokens_processed = args.batch_size * step_count * args.context_length
        final_step_count = step_count
        lr = cosine_lr_wup(
            i,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x.long())
        loss = cross_entropy(logits, y.long())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        should_eval = (i % args.eval_interval == 0) or (i == args.max_iters - 1)
        should_log_train = (
            ((i % args.train_log_interval == 0) or (i == args.max_iters - 1))
            and not should_eval
        )
        if should_log_train:
            elapsed = time.perf_counter() - t0
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/lr": float(lr),
                        "train/tokens_seen": int((i + 1) * tokens_per_step),
                        "train/total_tokens_processed": int(total_tokens_processed),
                        "train/wall_time_sec": float(elapsed),
                    },
                    step=i,
                )
            print(
                f"[train] iter={i:6d} | train_loss={loss.item():.4f} | "
                f"lr={lr:.2e} | tokens_seen={(i + 1) * tokens_per_step} | "
                f"total_tokens_processed={total_tokens_processed} | t={elapsed:.1f}s"
            )

        if should_eval:
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_batches=args.eval_batches,
            )
            elapsed = time.perf_counter() - t0
            record = {
                "run_name": args.run_name,
                "experiment": args.experiment,
                "step": i,
                "total_step_count": step_count,
                "tokens_seen": (i + 1) * tokens_per_step,
                "total_tokens_processed": total_tokens_processed,
                "wall_time_sec": elapsed,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "lr": lr,
                "batch_size": args.batch_size,
            }
            _append_log(args.log_path, record)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/val_loss": float(val_loss),
                        "eval/train_loss": float(loss.item()),
                        "eval/lr": float(lr),
                        "eval/wall_time_sec": float(elapsed),
                        "eval/total_tokens_processed": int(total_tokens_processed),
                    },
                    step=i,
                )
            print(
                f"iter={i:6d} | train_loss={loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | lr={lr:.2e} | "
                f"total_tokens_processed={total_tokens_processed} | t={elapsed:.1f}s"
            )

        should_ckpt = (i % args.checkpoint_interval == 0) or (i == args.max_iters - 1)
        if should_ckpt:
            save_checkpoint(model, optimizer, i + 1, args.checkpoint_path)
            getattr(output_volume, "commit", lambda: None)()
    print("done", args.log_path, args.batch_size * final_step_count * args.context_length)
    if wandb_run is not None:
        wandb_run.finish()


def _normalize_remote_device(device: str) -> str:
    d = device.lower()
    if d == "mps":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return d


@app.function(
    image=image,
    volumes={
        REMOTE_OUTPUT_DIR: output_volume,
        REMOTE_TOKENIZER_OUTPUT_DIR: tokenizer_output_volume,
    },
    timeout=24 * 60 * 60,
    gpu="B200",
)
def run_train_lm_remote(config: dict[str, Any] | None = None) -> str:
    os.chdir(REMOTE_WORKDIR)
    args = _namespace_from_config(config or {}, remote=True)
    train(args)
    output_volume.commit()
    return (
        f"Finished training run '{args.run_name}'. "
        f"Checkpoint: {args.checkpoint_path} | Log: {args.log_path}"
    )


@app.local_entrypoint()
def main(config_json: str = "") -> None:
    o = json.loads(config_json) if config_json.strip() else {}
    cfg = _resolve_train_config(o if isinstance(o, dict) else {})
    c = modal.Function.from_name(APP_NAME, "run_train_lm_remote").spawn(config=cfg)
    print(c.object_id, cfg["checkpoint_path"])


if __name__ == "__main__":
    main()