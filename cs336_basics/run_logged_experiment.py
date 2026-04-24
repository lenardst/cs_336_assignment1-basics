import json
import os
from pathlib import Path
from typing import Any

import modal

from cs336_basics import train_lm as lm

APP_NAME = "train-lm-logged"
WANDB_SECRET_NAME = "my-wandb-secret"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "einops", "wandb")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(lm.DATA_DIR), remote_path=lm.REMOTE_DATA_DIR)
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={
        lm.REMOTE_OUTPUT_DIR: lm.output_volume,
        lm.REMOTE_TOKENIZER_OUTPUT_DIR: lm.tokenizer_output_volume,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    timeout=24 * 60 * 60,
    gpu="B200",
)
def run_logged_experiment(config: dict[str, Any] | None = None) -> str:
    os.chdir(lm.REMOTE_WORKDIR)
    overrides = dict(config or {})
    overrides["enable_wandb"] = True
    if "model_variant" not in overrides:
        experiment = str(overrides.get("experiment", "baseline"))
        overrides["model_variant"] = lm.EXPERIMENT_TO_MODEL_VARIANT.get(experiment, "baseline")
    args = lm._namespace_from_config(overrides, remote=True)
    lm.train(args)
    lm.output_volume.commit()
    return (
        f"Finished run '{args.run_name}'. "
        f"Checkpoint: {args.checkpoint_path} | Log: {args.log_path}"
    )


@app.local_entrypoint()
def main(config_json: str = "") -> None:
    overrides: dict[str, Any] = {}
    if config_json.strip():
        overrides = json.loads(config_json)
        if not isinstance(overrides, dict):
            raise ValueError("config_json must decode to a JSON object.")
    overrides["enable_wandb"] = True

    try:
        deployed_fn = modal.Function.from_name(APP_NAME, "run_logged_experiment")
    except Exception as exc:
        raise RuntimeError(
            "Could not find deployed Modal function 'run_logged_experiment'. "
            f"Deploy first with: modal deploy {Path(__file__).as_posix()}"
        ) from exc

    function_call = deployed_fn.spawn(config=overrides)
    print("Submitted run_logged_experiment job asynchronously.")
    print(f"FunctionCall ID: {function_call.object_id}")
    print(
        f"Artifacts are written to Modal Volume '{lm.OUTPUT_VOLUME_NAME}' at "
        f"{lm.REMOTE_OUTPUT_DIR}: "
        f"{overrides.get('checkpoint_path', lm.TRAIN_CONFIG_DEFAULTS['checkpoint_path'])} and "
        f"{overrides.get('log_path', lm.TRAIN_CONFIG_DEFAULTS['log_path'])}"
    )

