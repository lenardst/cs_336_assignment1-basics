import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl_log(log_path: str | Path) -> list[dict]:
    path = Path(log_path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def plot_training_trajectory(
    records: list[dict], out_path: str | Path, x_key: str = "step"
) -> Path:
    x = [r.get(x_key) for r in records]
    train_loss = [r.get("train_loss") for r in records]
    val_loss = [r.get("val_loss") for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, train_loss, label="train_loss")
    ax.plot(x, val_loss, label="val_loss")
    ax.set_xlabel(x_key)
    ax.set_ylabel("loss")
    ax.set_title("Training trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_outcome(records: list[dict], out_path: str | Path) -> Path:
    wall = [r.get("wall_time_sec", 0.0) for r in records]
    val_loss = [r.get("val_loss") for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(wall, val_loss, marker="o", linewidth=1.5)
    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    ax.scatter([wall[best_idx]], [val_loss[best_idx]], s=60, label="best")
    ax.set_xlabel("wall_time_sec")
    ax.set_ylabel("val_loss")
    ax.set_title("Outcome vs wall clock")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CS336 training logs from JSONL.")
    parser.add_argument("--log", required=True, help="Path to train_log.jsonl")
    parser.add_argument(
        "--out-dir", default="plots", help="Directory where plot images are saved"
    )
    parser.add_argument(
        "--x-key",
        default="step",
        choices=["step", "total_step_count", "total_tokens_processed", "wall_time_sec"],
        help="X-axis used for trajectory plot",
    )
    args = parser.parse_args()

    records = load_jsonl_log(args.log)
    out_dir = Path(args.out_dir)
    trajectory_path = plot_training_trajectory(
        records, out_dir / "training_trajectory.png", x_key=args.x_key
    )
    outcome_path = plot_outcome(records, out_dir / "outcome_vs_wall_time.png")
    print(f"Saved: {trajectory_path}")
    print(f"Saved: {outcome_path}")


if __name__ == "__main__":
    main()

