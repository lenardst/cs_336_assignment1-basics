import time

import modal

APP, FN = "train-lm-logged", "run_logged_experiment"
app = modal.App("q7-layer-norm-ablation")


def _cfg(experiment: str, model_variant: str, lr: float, run_name: str, max_iters: int, batch_size: int):
    return {
        "experiment": experiment,
        "model_variant": model_variant,
        "run_name": run_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "context_length": 256,
        "checkpoint_path": f"checkpoints/{run_name}.pt",
        "log_path": f"logs/{run_name}.jsonl",
    }


def _tag(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


@app.local_entrypoint()
def main(
    base_learning_rate: float = 0.0028,
    lower_learning_rate: float = 0.0007,
    max_iters: int = 10_000,
    batch_size: int = 128,
) -> None:
    t = int(time.time())
    fn = modal.Function.from_name(APP, FN)
    cfgs = [
        _cfg("layer_norm_ablation", "baseline", base_learning_rate, f"q7_layernorm_baseline_lr{_tag(base_learning_rate)}_{t}", max_iters, batch_size),
        _cfg("layer_norm_ablation", "layer_norm_ablation", base_learning_rate, f"q7_layernorm_no_rmsnorm_lr{_tag(base_learning_rate)}_{t}", max_iters, batch_size),
        _cfg("layer_norm_ablation", "layer_norm_ablation", lower_learning_rate, f"q7_layernorm_no_rmsnorm_lr{_tag(lower_learning_rate)}_{t}", max_iters, batch_size),
    ]
    for c in cfgs:
        print(fn.spawn(config=c).object_id)
