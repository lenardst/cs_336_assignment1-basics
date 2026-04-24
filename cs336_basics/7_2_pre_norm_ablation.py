import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "train-lm-logged"
FUNCTION_NAME = "run_logged_experiment"
app = modal.App("q7-pre-norm-ablation")


def _run_tag(lr: float) -> str:
    return str(lr).replace("-", "m").replace(".", "p")


def _build_run_config(
    experiment: str,
    model_variant: str,
    lr: float,
    run_name: str,
    max_iters: int,
    batch_size: int,
) -> dict[str, Any]:
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


def _submit(configs: list[dict[str, Any]]) -> None:
    try:
        deployed_fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not find deployed Modal function '{FUNCTION_NAME}'. "
            f"Deploy first with: modal deploy {Path(__file__).as_posix()}"
        ) from exc

    for config in configs:
        call = deployed_fn.spawn(config=config)
        print(f"submitted run={config['run_name']} call_id={call.object_id}")


@app.local_entrypoint()
def main(
    learning_rate: float = 0.0028,
    max_iters: int = 10_000,
    batch_size: int = 128,
) -> None:
    stamp = int(time.time())
    configs = [
        _build_run_config(
            experiment="pre_norm_ablation",
            model_variant="baseline",
            lr=learning_rate,
            run_name=f"q7_prenorm_baseline_lr{_run_tag(learning_rate)}_{stamp}",
            max_iters=max_iters,
            batch_size=batch_size,
        ),
        _build_run_config(
            experiment="pre_norm_ablation",
            model_variant="pre_norm_ablation",
            lr=learning_rate,
            run_name=f"q7_prenorm_postnorm_lr{_run_tag(learning_rate)}_{stamp}",
            max_iters=max_iters,
            batch_size=batch_size,
        ),
    ]
    _submit(configs)
