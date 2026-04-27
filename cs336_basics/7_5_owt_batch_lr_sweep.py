import time

import modal

from cs336_basics import train_lm as lm

APP, FN = "train-lm-logged", "run_logged_experiment"
app = modal.App("q7-owt-batch-lr-sweep")

TRAIN = f"{lm.REMOTE_TOKENIZER_OUTPUT_DIR}/owt_train_uint16.npy"
VAL = f"{lm.REMOTE_TOKENIZER_OUTPUT_DIR}/owt_valid_uint16.npy"
_ITERS = {64: 20_000, 128: 10_000, 256: 5_000}
LRS = (1e-3, 3e-3, 1e-2, 3e-2)


def _cfg(run_name: str, lr: float, batch_size: int, max_iters: int, context_length: int):
    ev = max(1, max_iters)
    return {
        "experiment": "owt_batch_lr_sweep",
        "model_variant": "baseline",
        "run_name": run_name,
        "train_path": TRAIN,
        "val_path": VAL,
        "vocab_size": 32_000,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "context_length": context_length,
        "eval_interval": ev,
        "eval_batches": 3,
        "checkpoint_interval": max_iters * 1000,
        "checkpoint_path": f"checkpoints/{run_name}.pt",
        "log_path": f"logs/{run_name}.jsonl",
    }


def _tag(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


@app.local_entrypoint()
def main(context_length: int = 256) -> None:
    t = int(time.time())
    fn = modal.Function.from_name(APP, FN)
    for bs in (64, 128, 256):
        mi = _ITERS[bs]
        for lr in LRS:
            name = f"q7_owt_bs{bs}_lr{_tag(lr)}_it{mi}_{t}"
            print(fn.spawn(config=_cfg(name, lr, bs, mi, context_length)).object_id)
