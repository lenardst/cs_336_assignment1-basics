import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "train-lm-logged"
FUNCTION_NAME = "run_logged_experiment"
app = modal.App("q7-learning-rate-sweep")


def _run_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _build_run_config(
    run_name: str,
    learning_rate: float,
    batch_size: int,
    max_iters: int,
    context_length: int,
) -> dict[str, Any]:
    return {
        "experiment": "learning_rate",
        "model_variant": "baseline",
        "run_name": run_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "context_length": context_length,
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
    max_iters: int = 10_000,
    batch_size: int = 128,
    context_length: int = 256,
) -> None:
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]
    stamp = int(time.time())
    configs = []
    for lr in learning_rates:
        run_name = f"q7_learning_rate_bs{batch_size}_lr{_run_tag(lr)}_{stamp}"
        configs.append(
            _build_run_config(
                run_name=run_name,
                learning_rate=lr,
                batch_size=batch_size,
                max_iters=max_iters,
                context_length=context_length,
            )
        )
    _submit(configs)
