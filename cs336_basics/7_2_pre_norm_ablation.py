import time

import modal

APP, FN = "train-lm-logged", "run_logged_experiment"
app = modal.App("q7-pre-norm-ablation")


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
def main(learning_rate: float = 0.0028, max_iters: int = 10_000, batch_size: int = 128) -> None:
    t = int(time.time())
    fn = modal.Function.from_name(APP, FN)
    for c in (
        _cfg("pre_norm_ablation", "baseline", learning_rate, f"q7_prenorm_baseline_lr{_tag(learning_rate)}_{t}", max_iters, batch_size),
        _cfg("pre_norm_ablation", "pre_norm_ablation", learning_rate, f"q7_prenorm_postnorm_lr{_tag(learning_rate)}_{t}", max_iters, batch_size),
    ):
        print(fn.spawn(config=c).object_id)
