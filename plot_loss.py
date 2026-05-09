"""Plot training/validation loss CSV files written by main.py.

Usage:
    python plot_loss.py pos_loss.csv
    python plot_loss.py pos_loss.csv --output pos_loss.png
    python plot_loss.py pos_loss.csv --show
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def load_loss_csv(path):
    steps = []
    losses = []
    perplexities = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        required = {"step", "loss", "perplexity"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must have columns: step, loss, perplexity")

        for row in reader:
            if not row:
                continue
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            perplexities.append(float(row["perplexity"]))

    if not steps:
        raise ValueError(f"No data rows found in {path}")

    return steps, losses, perplexities


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss and perplexity from a main.py loss CSV."
    )
    parser.add_argument("loss_csv", help="Path to CSV with step, loss, perplexity columns")
    parser.add_argument("--output", "-o", default=None,
                        help="PNG path to write (default: <loss_csv stem>.png)")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window in addition to saving the plot")
    parser.add_argument("--log-perplexity", action="store_true",
                        help="Use a log scale for the perplexity axis")
    args = parser.parse_args()

    csv_path = Path(args.loss_csv)
    output_path = Path(args.output) if args.output else csv_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        steps, losses, perplexities = load_loss_csv(csv_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))
    try:
        if not args.show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib",
              file=sys.stderr)
        return 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(steps, losses, marker="o", linewidth=1.5, markersize=3)
    axes[0].set_title(csv_path.name)
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, perplexities, marker="o", linewidth=1.5, markersize=3,
                 color="tab:orange")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("perplexity")
    axes[1].grid(True, alpha=0.3)
    if args.log_perplexity:
        axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
