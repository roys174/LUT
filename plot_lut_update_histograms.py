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


def lut_group(name):
    if name.startswith("ffn_"):
        return "ffn"
    if name.startswith("head_"):
        if name.endswith(("_Q", "_K", "_V", "_Final")):
            return "new_attention"
        return "attention"
    return "other"


def lut_position(name):
    ffn_match = re.match(r"^ffn_(\d+)$", name)
    if ffn_match:
        return int(ffn_match.group(1)), None
    head_match = re.match(r"^head_(\d+)_(\d+)(?:_|$)", name)
    if head_match:
        return int(head_match.group(1)), int(head_match.group(2))
    return None, None


def parse_int_list(text, flag_name):
    values = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError as exc:
            raise ValueError(f"Bad {flag_name} value {text!r}; expected comma-separated integers") from exc
    return values


def lut_passes_filters(name, layers, heads, min_layer, max_layer):
    layer, head = lut_position(name)
    if layer is None:
        return True
    if layers is not None and layer not in layers:
        return False
    if min_layer is not None and layer < min_layer:
        return False
    if max_layer is not None and layer > max_layer:
        return False
    if head is not None and heads is not None and head not in heads:
        return False
    return True


def sample_table_indices(n_tables, sample_size, rng):
    if sample_size is None or sample_size < 1 or sample_size >= n_tables:
        return np.arange(n_tables, dtype=np.int32)
    return np.sort(rng.choice(n_tables, size=sample_size, replace=False).astype(np.int32))


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
                        help="Plot sampled tables for a LUT on one axis instead of one subplot per table")
    parser.add_argument("--ffn-tables", type=int, default=3,
                        help="Random FFN tables to plot per FFN LUT (default: 3; 0 = all)")
    parser.add_argument("--attention-tables", type=int, default=3,
                        help="Random attention tables to plot per attention LUT (default: 3; 0 = all)")
    parser.add_argument("--new-attention-tables", type=int, default=3,
                        help="Random new-attention Q/K/V/Final tables to plot per LUT (default: 3; 0 = all)")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for random table sampling (default: 0)")
    parser.add_argument("--include-other-luts", action="store_true",
                        help="Also plot non-FFN/non-attention LUTs, such as output heads")
    parser.add_argument("--layers", default=None,
                        help="Only plot these layer indices, comma-separated, e.g. 2,3")
    parser.add_argument("--heads", default=None,
                        help="Only plot these attention head indices, comma-separated, e.g. 0,2")
    parser.add_argument("--min-layer", type=int, default=None,
                        help="Only plot LUTs from this layer index or later")
    parser.add_argument("--max-layer", type=int, default=None,
                        help="Only plot LUTs up to this layer index")
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

    rng = np.random.default_rng(args.sample_seed)
    ffn_sample = None if args.ffn_tables == 0 else args.ffn_tables
    attention_sample = None if args.attention_tables == 0 else args.attention_tables
    new_attention_sample = None if args.new_attention_tables == 0 else args.new_attention_tables
    try:
        layer_filter = parse_int_list(args.layers, "--layers") if args.layers else None
        head_filter = parse_int_list(args.heads, "--heads") if args.heads else None
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    written = []
    for lut_name in sorted(stats):
        if not lut_passes_filters(lut_name, layer_filter, head_filter,
                                  args.min_layer, args.max_layer):
            continue
        counts = stats[lut_name]
        if counts.ndim != 2:
            print(f"Skipping {lut_name}: expected 2D array, got shape {counts.shape}",
                  file=sys.stderr)
            continue

        group = lut_group(lut_name)
        if group == "other" and not args.include_other_luts:
            continue

        n_tables, n_rows = counts.shape
        rows = np.arange(1, n_rows + 1)
        if group == "ffn":
            selected_tables = sample_table_indices(n_tables, ffn_sample, rng)
        elif group == "attention":
            selected_tables = sample_table_indices(n_tables, attention_sample, rng)
        elif group == "new_attention":
            selected_tables = sample_table_indices(n_tables, new_attention_sample, rng)
        else:
            selected_tables = np.arange(n_tables, dtype=np.int32)

        if selected_tables.size == 0:
            continue

        if args.overlay_tables:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            for table_idx in selected_tables:
                if args.line:
                    ax.plot(rows, counts[table_idx], linewidth=1.0,
                            label=f"table {table_idx}")
                else:
                    ax.bar(rows, counts[table_idx], width=1.0, alpha=0.35,
                           label=f"table {table_idx}")
            ax.set_title(f"{lut_name}: sampled updates per row")
            ax.set_xlabel("row")
            ax.set_ylabel("updates")
            ax.set_xlim(1, n_rows)
            ax.legend(loc="best")
        else:
            fig, axes = plt.subplots(selected_tables.size, 1,
                                     figsize=(12, max(3, 2.5 * selected_tables.size)),
                                     squeeze=False, sharex=True)
            for table_idx, ax in zip(selected_tables, axes[:, 0]):
                plot_table_rows(ax, counts[table_idx],
                                f"{lut_name} table {table_idx}: updates per row",
                                use_line=args.line)

        fig.tight_layout()
        output_path = out_dir / f"{_safe_filename(lut_name)}.png"
        fig.savefig(output_path, dpi=150)
        written.append(output_path)
        print(f"{lut_name}: plotted tables {', '.join(str(int(i)) for i in selected_tables)}")
        if args.show:
            plt.show()
        plt.close(fig)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
