import argparse
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch

transformer_module = importlib.import_module("cs336_basics.3_transformer_lm")
training_module = importlib.import_module("cs336_basics.4_training")

TransformerLM = getattr(transformer_module, "TransformerLM")
cross_entropy = getattr(training_module, "cross_entropy")
get_batch = getattr(training_module, "get_batch")
AdamW = getattr(training_module, "AdamW")
cosine_lr_wup = getattr(training_module, "cosine_lr_wup")
gradient_clipping = getattr(training_module, "gradient_clipping")
save_checkpoint = getattr(training_module, "save_checkpoint")
load_checkpoint = getattr(training_module, "load_checkpoint")

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DEFAULT_MODAL_DEVICE = "cuda"

# Assignment baseline defaults.
DEFAULTS = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1_344,
    "rope_theta": 10_000.0,
    "batch_size": 32,
    "max_iters": 40_000 if DEFAULT_DEVICE == "cuda" else 5_000,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_iters": 200,
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
    "norm_style": "pre",
    "pos_emb": "rope",
    "ffn_style": "swiglu",
}


def _resolve_train_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(TRAIN_CONFIG_DEFAULTS)
    if overrides:
        config.update(overrides)
    if not config["run_name"]:
        config["run_name"] = f"run_{int(time.time())}"
    return config


def load_token_array(path: Path) -> np.ndarray | np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r+")
    return np.memmap(path, dtype=np.uint16, mode="r+")


def load_train_val_arrays(train_path: Path, val_path: Path) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap]:
    return load_token_array(train_path), load_token_array(val_path)


def _ablation_overrides(args: argparse.Namespace) -> dict[str, object]:
    norm_style = args.norm_style
    pos_emb = args.pos_emb
    ffn_style = args.ffn_style

    if args.experiment == "layer_norm_ablation":
        norm_style = "none"
    elif args.experiment == "pre_norm_ablation":
        norm_style = "post"
    elif args.experiment == "no_pos_emb":
        pos_emb = "nope"
    elif args.experiment == "swiglu_ablation":
        ffn_style = "silu"

    requested: dict[str, object] = {}
    if norm_style == "none":
        requested["use_rmsnorm"] = False
    elif norm_style == "post":
        requested["norm_style"] = "post"

    if pos_emb == "nope":
        requested["use_rope"] = False
    if ffn_style == "silu":
        requested["ffn_style"] = "silu"
    return requested


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

    supported = set(inspect.signature(TransformerLM.__init__).parameters)
    supported.discard("self")
    for key, value in _ablation_overrides(args).items():
        if key not in supported:
            raise ValueError(
                f"Experiment needs TransformerLM.__init__ argument '{key}', but it is not implemented yet."
            )
        model_kwargs[key] = value

    model = TransformerLM(**model_kwargs)
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
    print(
        f"[startup] run={args.run_name} experiment={args.experiment} "
        f"device={args.device} dtype={args.dtype} max_iters={args.max_iters}"
    )
    print(f"[data] loading train array from {args.train_path}")
    train_data, val_data = load_train_val_arrays(args.train_path, args.val_path)
    print(
        f"[data] loaded train shape={train_data.shape} dtype={train_data.dtype} "
        f"val shape={val_data.shape} dtype={val_data.dtype}"
    )
    print(
        f"[data] batch_size={args.batch_size} context_length={args.context_length} "
        f"tokens_per_step={args.batch_size * args.context_length}"
    )
    print("[model] building TransformerLM")
    model = _build_model(args)
    num_params = sum(param.numel() for param in model.parameters())
    print(f"[model] built model with parameters={num_params:,}")
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
        print(f"[checkpoint] loading checkpoint from {args.checkpoint_path}")
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"[checkpoint] resumed at iter={start_iter}")
    else:
        print(f"[checkpoint] starting fresh; output path={args.checkpoint_path}")

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
        print(f"[wandb] initialized run={wandb_run.id}")
    print("[train] entering optimization loop")
    for i in range(start_iter, args.max_iters):
        step_count = i + 1
        total_tokens_processed = args.batch_size * step_count * args.context_length
        final_step_count = step_count
        try:
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
        except Exception as exc:
            raise RuntimeError(
                f"Training failed at iter={i} "
                f"(device={args.device}, batch_size={args.batch_size}, context_length={args.context_length})"
            ) from exc

        should_log_train = (i % args.train_log_interval == 0) or (i == args.max_iters - 1)
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

        should_eval = (i % args.eval_interval == 0) or (i == args.max_iters - 1)
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
            print(f"[checkpoint] saved iter={i + 1} -> {args.checkpoint_path}")
            try:
                output_volume.commit()
            except Exception:
                # Local non-Modal runs should not fail on volume commit.
                pass
    final_total_tokens_processed = args.batch_size * final_step_count * args.context_length
    print(
        f"[done] training complete, logs at {args.log_path} | "
        f"total_tokens_processed={final_total_tokens_processed}"
    )
    if wandb_run is not None:
        wandb_run.finish()


def _normalize_remote_device(device: str) -> str:
    normalized = device.lower()
    if normalized == "mps":
        if torch.cuda.is_available():
            print("[startup] requested device=mps in Modal; remapping to cuda")
            return "cuda"
        print("[startup] requested device=mps in Modal; remapping to cpu")
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print("[startup] requested device=cuda but CUDA unavailable; remapping to cpu")
        return "cpu"
    return normalized


@app.function(
    image=image,
    volumes={
        REMOTE_OUTPUT_DIR: output_volume,
        REMOTE_TOKENIZER_OUTPUT_DIR: tokenizer_output_volume,
    },
    timeout=24 * 60 * 60,
    gpu="A10G",
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
    try:
        deployed_fn = modal.Function.from_name(APP_NAME, "run_train_lm_remote")
    except Exception as exc:
        raise RuntimeError(
            "Could not find deployed Modal function 'run_train_lm_remote'. "
            f"Deploy first with: modal deploy {Path(__file__).as_posix()}"
        ) from exc

    overrides: dict[str, Any] = {}
    if config_json.strip():
        overrides = json.loads(config_json)
        if not isinstance(overrides, dict):
            raise ValueError("config_json must decode to a JSON object.")
    config = _resolve_train_config(overrides)
    function_call = deployed_fn.spawn(config=config)
    print("Submitted run_train_lm_remote job asynchronously.")
    print(f"FunctionCall ID: {function_call.object_id}")
    print(
        f"Artifacts will be written to Modal Volume '{OUTPUT_VOLUME_NAME}' at "
        f"{REMOTE_OUTPUT_DIR}: {config['checkpoint_path']} and {config['log_path']}"
    )


if __name__ == "__main__":
    main()