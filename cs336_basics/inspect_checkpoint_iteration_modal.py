import modal
import torch


APP_NAME = "inspect-checkpoint-iteration"
VOLUME_NAME = "cs336-lm-training"
REMOTE_MOUNT = "/lm_training_outputs"

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")
app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(image=image, volumes={REMOTE_MOUNT: output_volume})
def read_iteration(checkpoint_path: str) -> int:
    checkpoint = torch.load(f"{REMOTE_MOUNT}/{checkpoint_path}", map_location="cpu")
    return int(checkpoint.get("iteration", -1))


@app.local_entrypoint()
def main(checkpoint_path: str) -> None:
    iteration = read_iteration.remote(checkpoint_path)
    print(f"checkpoint={checkpoint_path}")
    print(f"iteration={iteration}")
