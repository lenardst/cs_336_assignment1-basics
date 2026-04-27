import time

import modal

APP, FN = "train-lm-logged", "run_logged_experiment"
app = modal.App("q7-learning-rate-sweep")


def _cfg(run_name: str, lr: float, batch_size: int, max_iters: int, context_length: int):
    return {
        "experiment": "learning_rate",
        "model_variant": "baseline",
        "run_name": run_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "context_length": context_length,
        "checkpoint_path": f"checkpoints/{run_name}.pt",
        "log_path": f"logs/{run_name}.jsonl",
    }


def _tag(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


@app.local_entrypoint()
def main(max_iters: int = 10_000, batch_size: int = 128, context_length: int = 256) -> None:
    t = int(time.time())
    fn = modal.Function.from_name(APP, FN)
    for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0):
        name = f"q7_learning_rate_bs{batch_size}_lr{_tag(lr)}_{t}"
        print(fn.spawn(config=_cfg(name, lr, batch_size, max_iters, context_length)).object_id)
