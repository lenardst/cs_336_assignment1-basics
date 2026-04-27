import json
import os
from typing import Any

import modal

from cs336_basics import train_lm as lm

APP = "train-lm-logged"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "einops", "wandb")
    .add_local_python_source("cs336_basics")
    .add_local_dir(str(lm.DATA_DIR), remote_path=lm.REMOTE_DATA_DIR)
)
app = modal.App(APP)


@app.function(
    image=image,
    volumes={
        lm.REMOTE_OUTPUT_DIR: lm.output_volume,
        lm.REMOTE_TOKENIZER_OUTPUT_DIR: lm.tokenizer_output_volume,
    },
    secrets=[modal.Secret.from_name("my-wandb-secret")],
    timeout=24 * 60 * 60,
    gpu="B200",
)
def run_logged_experiment(config: dict[str, Any] | None = None) -> str:
    os.chdir(lm.REMOTE_WORKDIR)
    o = dict(config or {})
    o["enable_wandb"] = True
    if "model_variant" not in o:
        o["model_variant"] = lm.EXPERIMENT_TO_MODEL_VARIANT.get(str(o.get("experiment", "baseline")), "baseline")
    args = lm._namespace_from_config(o, remote=True)
    lm.train(args)
    lm.output_volume.commit()
    return f"Finished '{args.run_name}' | ckpt={args.checkpoint_path} | log={args.log_path}"


@app.local_entrypoint()
def main(config_json: str = "") -> None:
    o: dict[str, Any] = json.loads(config_json) if config_json.strip() else {}
    o["enable_wandb"] = True
    c = modal.Function.from_name(APP, "run_logged_experiment").spawn(config=o)
    print("spawned", c.object_id)
