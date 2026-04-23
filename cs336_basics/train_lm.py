import argparse
import importlib
import inspect
import json
import time
from pathlib import Path

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
    "checkpoint_interval": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransformerLM on TinyStories token IDs.")
    parser.add_argument(
        "--experiment",
        choices=[
            "baseline",
            "learning_rate",
            "batch_size",
            "layer_norm_ablation",
            "pre_norm_ablation",
            "no_pos_emb",
            "swiglu_ablation",
        ],
        default="baseline",
    )
    parser.add_argument("--run-name", default=f"run_{int(time.time())}")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/tokenizer_experiments/tiny_train_uint16.npy"),
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/tokenizer_experiments/tiny_valid_uint16.npy"),
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")

    parser.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    parser.add_argument("--context-length", type=int, default=DEFAULTS["context_length"])
    parser.add_argument("--d-model", type=int, default=DEFAULTS["d_model"])
    parser.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    parser.add_argument("--num-heads", type=int, default=DEFAULTS["num_heads"])
    parser.add_argument("--d-ff", type=int, default=DEFAULTS["d_ff"])
    parser.add_argument("--rope-theta", type=float, default=DEFAULTS["rope_theta"])

    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--max-iters", type=int, default=DEFAULTS["max_iters"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULTS["min_lr"])
    parser.add_argument("--warmup-iters", type=int, default=DEFAULTS["warmup_iters"])
    parser.add_argument("--beta1", type=float, default=DEFAULTS["beta1"])
    parser.add_argument("--beta2", type=float, default=DEFAULTS["beta2"])
    parser.add_argument("--epsilon", type=float, default=DEFAULTS["epsilon"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULTS["max_grad_norm"])

    parser.add_argument("--eval-interval", type=int, default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval-batches", type=int, default=DEFAULTS["eval_batches"])
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULTS["checkpoint_interval"])
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/lm.pt"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/train_log.jsonl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", choices=["off", "default", "aot_eager"], default="off")

    # Architecture ablation flags (these require model support in TransformerLM.__init__).
    parser.add_argument("--norm-style", choices=["pre", "post", "none"], default="pre")
    parser.add_argument("--pos-emb", choices=["rope", "nope"], default="rope")
    parser.add_argument("--ffn-style", choices=["swiglu", "silu"], default="swiglu")
    return parser.parse_args()


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


def train(args: argparse.Namespace) -> None:
    train_data, val_data = load_train_val_arrays(args.train_path, args.val_path)
    model = _build_model(args)
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
        print(f"Resumed checkpoint at iter={start_iter}")

    t0 = time.perf_counter()
    tokens_per_step = args.batch_size * args.context_length
    model.train()
    for i in range(start_iter, args.max_iters):
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
                "tokens_seen": (i + 1) * tokens_per_step,
                "wall_time_sec": elapsed,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "lr": lr,
                "batch_size": args.batch_size,
            }
            _append_log(args.log_path, record)
            print(
                f"iter={i:6d} | train_loss={loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | lr={lr:.2e} | t={elapsed:.1f}s"
            )

        should_ckpt = (i % args.checkpoint_interval == 0) or (i == args.max_iters - 1)
        if should_ckpt:
            save_checkpoint(model, optimizer, i + 1, args.checkpoint_path)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()