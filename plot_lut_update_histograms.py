"""Plot row-by-row LUT update counts produced by main.py.

Usage:
    python plot_lut_update_histograms.py lut_update_stats.npz

The input must be the raw .npz stats file written by --lut-update-stats-file.
Each array in that file has shape (n_tables, n_rows), where:
    X axis = row index, 1..n_rows
    Y axis = number of updates applied to that row
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


def _safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "lut"


def load_stats(path):
    stats_path = Path(path)
    if stats_path.suffix != ".npz":
        raise ValueError(
            "Row-by-row plots require the raw .npz stats file, not the "
            "_histogram.csv summary. Pass the path used with "
            "--lut-update-stats-file."
        )
    with np.load(stats_path) as data:
        return {name: data[name].copy() for name in data.files}


def plot_table_rows(ax, counts, title, use_line=False):
    rows = np.arange(1, counts.shape[0] + 1)
    if use_line:
        ax.plot(rows, counts, linewidth=1.0)
    else:
        ax.bar(rows, counts, width=1.0)
    ax.set_title(title)
    ax.set_xlabel("row")
    ax.set_ylabel("updates")
    ax.set_xlim(1, counts.shape[0])


def main():
    parser = argparse.ArgumentParser(
        description="Plot LUT row index vs number of updates from raw stats .npz."
    )
    parser.add_argument("stats_npz", help="Path to raw LUT update stats .npz")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for PNG plots (default: <npz stem>_row_plots next to input)")
    parser.add_argument("--line", action="store_true",
                        help="Use a line plot instead of bars")
    parser.add_argument("--overlay-tables", action="store_true",
                        help="Plot all tables for a LUT on one axis instead of one subplot per table")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive plot windows instead of only writing PNG files")
    args = parser.parse_args()

    stats_path = Path(args.stats_npz)
    out_dir = Path(args.out_dir) if args.out_dir else stats_path.with_suffix("").parent / (stats_path.stem + "_row_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        stats = load_stats(stats_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not stats:
        print(f"No arrays found in {stats_path}", file=sys.stderr)
        return 1

    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    try:
        if not args.show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib",
              file=sys.stderr)
        return 1

    written = []
    for lut_name in sorted(stats):
        counts = stats[lut_name]
        if counts.ndim != 2:
            print(f"Skipping {lut_name}: expected 2D array, got shape {counts.shape}",
                  file=sys.stderr)
            continue

        n_tables, n_rows = counts.shape
        rows = np.arange(1, n_rows + 1)

        if args.overlay_tables:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            for table_idx in range(n_tables):
                if args.line:
                    ax.plot(rows, counts[table_idx], linewidth=1.0,
                            label=f"table {table_idx}")
                else:
                    ax.bar(rows, counts[table_idx], width=1.0, alpha=0.35,
                           label=f"table {table_idx}")
            ax.set_title(f"{lut_name}: updates per row")
            ax.set_xlabel("row")
            ax.set_ylabel("updates")
            ax.set_xlim(1, n_rows)
            ax.legend(loc="best")
        else:
            fig, axes = plt.subplots(n_tables, 1, figsize=(12, max(3, 2.5 * n_tables)),
                                     squeeze=False, sharex=True)
            for table_idx, ax in enumerate(axes[:, 0]):
                plot_table_rows(ax, counts[table_idx],
                                f"{lut_name} table {table_idx}: updates per row",
                                use_line=args.line)

        fig.tight_layout()
        output_path = out_dir / f"{_safe_filename(lut_name)}.png"
        fig.savefig(output_path, dpi=150)
        written.append(output_path)
        if args.show:
            plt.show()
        plt.close(fig)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
