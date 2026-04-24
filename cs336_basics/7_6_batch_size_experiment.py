import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "train-lm-logged"
FUNCTION_NAME = "run_logged_experiment"
app = modal.App("q7-batch-size-experiment")


def _run_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _iters_for_batch_size(batch_size: int) -> int:
    if batch_size == 1:
        return 1_280_000
    if batch_size == 64:
        return 20_000
    if batch_size == 128:
        return 10_000
    if batch_size == 256:
        return 5_000
    if batch_size == 1024:
        return 1_250
    raise ValueError(f"Unsupported batch size: {batch_size}")


def _build_run_config(
    run_name: str,
    learning_rate: float,
    batch_size: int,
    max_iters: int,
    context_length: int,
) -> dict[str, Any]:
    return {
        "experiment": "batch_size_experiment",
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
def main(context_length: int = 256) -> None:
    batch_sizes = [1, 64, 128, 256, 1024]
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]
    stamp = int(time.time())
    configs = []

    for batch_size in batch_sizes:
        max_iters = _iters_for_batch_size(batch_size)
        for lr in learning_rates:
            run_name = (
                f"q7_batch_size_bs{batch_size}_lr{_run_tag(lr)}_iters{max_iters}_{stamp}"
            )
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
