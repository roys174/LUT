"""Analyze which LUT row indices receive most updates.

Example:
    python analyze_lut_hot_rows.py results/POS/20260601-185650/lut_update_stats.npz

By default this mirrors plot_lut_update_histograms.py: it samples 3 FFN tables
and 3 attention tables per LUT using a deterministic seed. Use --tables to
pin exact tables, for example:

    python analyze_lut_hot_rows.py results/POS/20260601-185650/lut_update_stats.npz \
        --tables head_0_0_Score:16,41,76
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


DEFAULT_STATS = "results/POS/20260601-185650/lut_update_stats.npz"


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "lut"


def resolve_stats_path(path):
    stats_path = Path(path)
    if stats_path.exists():
        return stats_path
    repo_relative = Path("LUT/LUT") / stats_path
    if repo_relative.exists():
        return repo_relative
    raise FileNotFoundError(path)


def load_stats(path):
    stats_path = resolve_stats_path(path)
    if stats_path.suffix != ".npz":
        raise ValueError("Expected the raw .npz file written by --lut-update-stats-file.")
    with np.load(stats_path) as data:
        return stats_path, {name: data[name].copy() for name in data.files}


def lut_group(name):
    if name.startswith("ffn_"):
        return "ffn"
    if name.startswith("head_"):
        return "attention"
    return "other"


def sample_table_indices(n_tables, sample_size, rng):
    if sample_size is None or sample_size < 1 or sample_size >= n_tables:
        return np.arange(n_tables, dtype=np.int32)
    return np.sort(rng.choice(n_tables, size=sample_size, replace=False).astype(np.int32))


def parse_table_specs(specs):
    selected = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"Bad --tables value {spec!r}; expected LUT_NAME:1,2,3")
        lut_name, table_text = spec.split(":", 1)
        tables = []
        for part in table_text.split(","):
            part = part.strip()
            if part:
                tables.append(int(part))
        if not tables:
            raise ValueError(f"Bad --tables value {spec!r}; no table indices were provided")
        selected.setdefault(lut_name, set()).update(tables)
    return {name: np.array(sorted(tables), dtype=np.int32)
            for name, tables in selected.items()}


def infer_row_layout(lut_name, n_rows, n_c, positional_dim):
    simple_rows = 1 << n_c
    attention_rows = 1 << (2 * n_c + positional_dim)
    if n_rows == attention_rows and (lut_name.startswith("head_") and
                                     (lut_name.endswith("_Score") or lut_name.endswith("_V"))):
        return "attention"
    if n_rows == simple_rows:
        return "simple"
    return "unknown"


def bit_string(value, width):
    return format(int(value), f"0{width}b")


def decode_row(lut_name, row, n_rows, n_c, positional_dim):
    layout = infer_row_layout(lut_name, n_rows, n_c, positional_dim)
    if layout == "attention":
        pe_mask = (1 << positional_dim) - 1
        k_mask = (1 << n_c) - 1
        pe = row & pe_mask
        k = (row >> positional_dim) & k_mask
        q = row >> (positional_dim + n_c)
        return {
            "layout": layout,
            "row_bits": bit_string(row, 2 * n_c + positional_dim),
            "q_bin": q,
            "k_bin": k,
            "pe_bin": pe,
            "q_bits": bit_string(q, n_c),
            "k_bits": bit_string(k, n_c),
            "pe_bits": bit_string(pe, positional_dim),
            "q_popcount": int(q).bit_count(),
            "k_popcount": int(k).bit_count(),
            "pe_popcount": int(pe).bit_count(),
        }
    if layout == "simple":
        return {
            "layout": layout,
            "row_bits": bit_string(row, n_c),
            "q_bin": "",
            "k_bin": "",
            "pe_bin": "",
            "q_bits": "",
            "k_bits": "",
            "pe_bits": "",
            "q_popcount": "",
            "k_popcount": "",
            "pe_popcount": "",
        }
    width = max(1, int(math.ceil(math.log2(max(2, n_rows)))))
    return {
        "layout": layout,
        "row_bits": bit_string(row, width),
        "q_bin": "",
        "k_bin": "",
        "pe_bin": "",
        "q_bits": "",
        "k_bits": "",
        "pe_bits": "",
        "q_popcount": "",
        "k_popcount": "",
        "pe_popcount": "",
    }


def concentration_stats(counts):
    counts_f = counts.astype(np.float64)
    total = counts_f.sum()
    nonzero = int(np.count_nonzero(counts_f))
    if total <= 0:
        return {
            "total_updates": 0,
            "nonzero_rows": nonzero,
            "max_updates": 0,
            "max_share": 0.0,
            "top3_share": 0.0,
            "top10_share": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "gini": 0.0,
        }

    sorted_counts = np.sort(counts_f)[::-1]
    probs = counts_f[counts_f > 0] / total
    entropy = float(-(probs * np.log2(probs)).sum())
    normalized_entropy = entropy / math.log2(counts_f.size) if counts_f.size > 1 else 0.0
    ascending = np.sort(counts_f)
    gini_num = np.sum((2 * np.arange(1, counts_f.size + 1) - counts_f.size - 1) * ascending)
    gini = float(gini_num / (counts_f.size * total))
    return {
        "total_updates": int(total),
        "nonzero_rows": nonzero,
        "max_updates": int(sorted_counts[0]),
        "max_share": float(sorted_counts[0] / total),
        "top3_share": float(sorted_counts[:3].sum() / total),
        "top10_share": float(sorted_counts[:10].sum() / total),
        "entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "gini": gini,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def zero_update_ranges(counts):
    """Return inclusive zero-count ranges as (start0, end0) pairs."""
    ranges = []
    start = None
    for idx, value in enumerate(counts):
        if value == 0:
            if start is None:
                start = idx
        elif start is not None:
            ranges.append((start, idx - 1))
            start = None
    if start is not None:
        ranges.append((start, counts.size - 1))
    return ranges


def format_ranges(ranges, limit=12):
    if not ranges:
        return "none"
    parts = []
    for start0, end0 in ranges[:limit]:
        start1 = start0 + 1
        end1 = end0 + 1
        if start1 == end1:
            parts.append(str(start1))
        else:
            parts.append(f"{start1}-{end1}")
    if len(ranges) > limit:
        parts.append(f"... (+{len(ranges) - limit} more)")
    return ", ".join(parts)


def add_zero_ranges(lut_name, table_idx, counts, n_c, positional_dim, rows):
    table_label = int(table_idx) if isinstance(table_idx, (int, np.integer)) else str(table_idx)
    for rank, (start0, end0) in enumerate(zero_update_ranges(counts), start=1):
        start_decoded = decode_row(lut_name, int(start0), counts.size, n_c, positional_dim)
        end_decoded = decode_row(lut_name, int(end0), counts.size, n_c, positional_dim)
        rows.append({
            "lut": lut_name,
            "table": table_label,
            "rank": rank,
            "start_row0": int(start0),
            "end_row0": int(end0),
            "start_row1": int(start0) + 1,
            "end_row1": int(end0) + 1,
            "length": int(end0 - start0 + 1),
            "layout": start_decoded["layout"],
            "start_bits": start_decoded["row_bits"],
            "end_bits": end_decoded["row_bits"],
            "start_q_bin": start_decoded["q_bin"],
            "end_q_bin": end_decoded["q_bin"],
            "start_k_bin": start_decoded["k_bin"],
            "end_k_bin": end_decoded["k_bin"],
            "start_pe_bin": start_decoded["pe_bin"],
            "end_pe_bin": end_decoded["pe_bin"],
        })

def selected_tables_for_stats(stats, explicit, ffn_sample, attention_sample, include_other, rng):
    if explicit:
        return explicit

    selected = {}
    for lut_name in sorted(stats):
        group = lut_group(lut_name)
        if group == "other" and not include_other:
            continue
        n_tables = stats[lut_name].shape[0]
        if group == "ffn":
            tables = sample_table_indices(n_tables, ffn_sample, rng)
        elif group == "attention":
            tables = sample_table_indices(n_tables, attention_sample, rng)
        else:
            tables = np.arange(n_tables, dtype=np.int32)
        selected[lut_name] = tables
    return selected


def add_top_rows(lut_name, table_idx, counts, top_n, n_c, positional_dim, rows):
    total = counts.sum()
    mean = counts.mean()
    std = counts.std()
    order = np.argsort(counts)[::-1][:top_n]
    table_label = int(table_idx) if isinstance(table_idx, (int, np.integer)) else str(table_idx)
    for rank, row_idx in enumerate(order, start=1):
        decoded = decode_row(lut_name, int(row_idx), counts.size, n_c, positional_dim)
        update_count = int(counts[row_idx])
        record = {
            "lut": lut_name,
            "table": table_label,
            "rank": rank,
            "row0": int(row_idx),
            "row1": int(row_idx) + 1,
            "updates": update_count,
            "share": float(update_count / total) if total else 0.0,
            "zscore": float((update_count - mean) / std) if std > 0 else 0.0,
        }
        record.update(decoded)
        rows.append(record)


def add_bit_bias(lut_name, table_idx, counts, n_c, positional_dim, rows):
    layout = infer_row_layout(lut_name, counts.size, n_c, positional_dim)
    total = counts.sum()
    width = n_c if layout == "simple" else 2 * n_c + positional_dim
    if layout == "unknown" or total == 0:
        return
    row_indices = np.arange(counts.size, dtype=np.int64)
    for bit in range(width):
        bit_mask = (row_indices & (1 << bit)) != 0
        updates_with_bit = int(counts[bit_mask].sum())
        if layout == "attention":
            if bit < positional_dim:
                component = "PE"
                component_bit = bit
            elif bit < positional_dim + n_c:
                component = "K"
                component_bit = bit - positional_dim
            else:
                component = "Q"
                component_bit = bit - positional_dim - n_c
        else:
            component = "simple"
            component_bit = bit
        rows.append({
            "lut": lut_name,
            "table": int(table_idx),
            "layout": layout,
            "bit": bit,
            "component": component,
            "component_bit": component_bit,
            "updates_with_bit_1": updates_with_bit,
            "share_with_bit_1": float(updates_with_bit / total),
        })


def add_modulo_patterns(lut_name, table_idx, counts, rows):
    for modulo in [2, 4, 8, 16, 32, 64]:
        if modulo > counts.size:
            continue
        residue_counts = np.zeros(modulo, dtype=np.uint64)
        for residue in range(modulo):
            residue_counts[residue] = counts[residue::modulo].sum()
        total = residue_counts.sum()
        order = np.argsort(residue_counts)[::-1]
        for rank, residue in enumerate(order[:min(8, modulo)], start=1):
            updates = int(residue_counts[residue])
            rows.append({
                "lut": lut_name,
                "table": int(table_idx),
                "modulo": modulo,
                "rank": rank,
                "residue0": int(residue),
                "residue1": int(residue) + 1,
                "updates": updates,
                "share": float(updates / total) if total else 0.0,
            })


def component_marginals(lut_name, table_idx, counts, n_c, positional_dim):
    layout = infer_row_layout(lut_name, counts.size, n_c, positional_dim)
    if layout != "attention":
        return []
    q_size = 1 << n_c
    k_size = 1 << n_c
    pe_size = 1 << positional_dim
    cube = counts.reshape(q_size, k_size, pe_size)
    rows = []
    for component, values in [
        ("Q", cube.sum(axis=(1, 2))),
        ("K", cube.sum(axis=(0, 2))),
        ("PE", cube.sum(axis=(0, 1))),
    ]:
        total = values.sum()
        for value, updates in enumerate(values):
            rows.append({
                "lut": lut_name,
                "table": int(table_idx),
                "component": component,
                "value0": int(value),
                "value1": int(value) + 1,
                "bits": bit_string(value, n_c if component != "PE" else positional_dim),
                "updates": int(updates),
                "share": float(updates / total) if total else 0.0,
            })
    return rows


def make_plots(out_dir, analyses, n_c, positional_dim, use_line, show=False, backend=None):
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    import matplotlib
    if backend:
        matplotlib.use(backend, force=True)
    elif not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if show:
        backend_name = plt.get_backend()
        print(f"Matplotlib backend: {backend_name}")
        if backend_name.lower().endswith("agg"):
            print(
                "--show requested, but this Matplotlib backend cannot open a window. "
                "Try --mpl-backend TkAgg or --mpl-backend QtAgg from a session with a display.",
                file=sys.stderr,
            )

    for lut_name, table_idx, counts in analyses:
        rows = np.arange(1, counts.size + 1)
        top = np.argsort(counts)[::-1][:min(12, counts.size)]
        zero_ranges = zero_update_ranges(counts)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        zero_label_used = False
        for start0, end0 in zero_ranges:
            ax.axvspan(start0 + 0.5, end0 + 1.5, color="0.88", alpha=0.8,
                       label="0-update range" if not zero_label_used else None,
                       zorder=0)
            zero_label_used = True

        if use_line:
            ax.plot(rows, counts, linewidth=1.0)
        else:
            ax.bar(rows, counts, width=1.0)
        ax.scatter(top + 1, counts[top], color="crimson", s=18, zorder=3)
        for row_idx in top[:6]:
            ax.annotate(str(int(row_idx) + 1), (int(row_idx) + 1, counts[row_idx]),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=8)

        ymax = max(float(counts.max()), 1.0)
        if zero_ranges and len(zero_ranges) <= 12:
            for start0, end0 in zero_ranges:
                start1 = start0 + 1
                end1 = end0 + 1
                label = str(start1) if start1 == end1 else f"{start1}-{end1}"
                ax.text((start1 + end1) * 0.5, ymax * 0.97, label,
                        ha="center", va="top", fontsize=7, rotation=90, color="0.25")
        elif zero_ranges:
            ax.text(0.995, 0.97, f"{len(zero_ranges)} zero-update ranges; see CSV",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85})

        ax.set_title(f"{lut_name} table {table_idx}: hot update rows")
        ax.set_xlabel("row (1-based)")
        ax.set_ylabel("updates")
        if zero_ranges:
            ax.legend(loc="upper left")
        fig.tight_layout()
        if show:
            plt.show(block=True)
        else:
            fig.savefig(out_dir / f"{safe_filename(lut_name)}_table_{table_idx}_hot_rows.png", dpi=150)
        plt.close(fig)

        layout = infer_row_layout(lut_name, counts.size, n_c, positional_dim)
        if layout != "attention":
            continue
        q_size = 1 << n_c
        k_size = 1 << n_c
        pe_size = 1 << positional_dim
        cube = counts.reshape(q_size, k_size, pe_size)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        qk = cube.sum(axis=2)
        im = axes[0].imshow(qk, cmap="viridis", aspect="auto")
        axes[0].set_title("Q x K, summed over PE")
        axes[0].set_xlabel("K bin")
        axes[0].set_ylabel("Q bin")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        pe_values = cube.sum(axis=(0, 1))
        axes[1].bar(np.arange(pe_size), pe_values)
        axes[1].set_title("PE marginal")
        axes[1].set_xlabel("PE bin (0-based)")
        axes[1].set_ylabel("updates")

        q_values = cube.sum(axis=(1, 2))
        k_values = cube.sum(axis=(0, 2))
        x = np.arange(q_size)
        axes[2].bar(x - 0.15, q_values, width=0.3, label="Q")
        axes[2].bar(x + 0.15, k_values, width=0.3, label="K")
        axes[2].set_title("Q/K marginals")
        axes[2].set_xlabel("bin (0-based)")
        axes[2].set_ylabel("updates")
        axes[2].legend()
        fig.suptitle(f"{lut_name} table {table_idx}: decoded attention structure")
        fig.tight_layout()
        if show:
            plt.show(block=True)
        else:
            fig.savefig(out_dir / f"{safe_filename(lut_name)}_table_{table_idx}_decoded.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hot LUT row indices from lut_update_stats.npz."
    )
    parser.add_argument("stats_npz", nargs="?", default=DEFAULT_STATS,
                        help=f"Raw stats .npz (default: {DEFAULT_STATS})")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <npz stem>_hot_row_analysis)")
    parser.add_argument("--tables", action="append", default=[],
                        help="Analyze exact tables, e.g. --tables head_0_0_Score:16,41,76")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top rows to report per selected table (default: 20)")
    parser.add_argument("--n-c", type=int, default=2,
                        help="LUT comparison bits per Q/K/simple LUT in this run (default: 2)")
    parser.add_argument("--positional-dim", type=int, default=4,
                        help="Positional bits in attention row indices (default: 4)")
    parser.add_argument("--ffn-tables", type=int, default=3,
                        help="Random FFN tables to analyze when --tables is omitted (default: 3; 0 = all)")
    parser.add_argument("--attention-tables", type=int, default=3,
                        help="Random attention tables to analyze when --tables is omitted (default: 3; 0 = all)")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for deterministic table sampling (default: 0)")
    parser.add_argument("--include-other-luts", action="store_true",
                        help="Also analyze output/other LUTs when --tables is omitted")
    parser.add_argument("--line", action="store_true",
                        help="Use line plots instead of bars for row-count plots")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plot windows instead of saving PNG files")
    parser.add_argument("--mpl-backend", default=None,
                        help="Matplotlib backend to use with --show, e.g. TkAgg or QtAgg")
    args = parser.parse_args()

    try:
        stats_path, stats = load_stats(args.stats_npz)
        explicit_tables = parse_table_specs(args.tables)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else stats_path.with_suffix("").parent / (stats_path.stem + "_hot_row_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.sample_seed)
    ffn_sample = None if args.ffn_tables == 0 else args.ffn_tables
    attention_sample = None if args.attention_tables == 0 else args.attention_tables
    selected = selected_tables_for_stats(
        stats, explicit_tables, ffn_sample, attention_sample, args.include_other_luts, rng
    )

    analyses = []
    summary_rows = []
    top_rows = []
    bit_bias_rows = []
    modulo_rows = []
    marginal_rows = []
    zero_range_rows = []
    zero_ranges_by_table = {}
    aggregate_by_lut = {}

    for lut_name in sorted(selected):
        if lut_name not in stats:
            print(f"Skipping {lut_name}: not found in {stats_path}", file=sys.stderr)
            continue
        counts_2d = stats[lut_name]
        if counts_2d.ndim != 2:
            print(f"Skipping {lut_name}: expected 2D array, got {counts_2d.shape}",
                  file=sys.stderr)
            continue
        n_tables = counts_2d.shape[0]
        valid_tables = [int(t) for t in selected[lut_name] if 0 <= int(t) < n_tables]
        if len(valid_tables) != len(selected[lut_name]):
            print(f"{lut_name}: ignored table indices outside [0, {n_tables - 1}]",
                  file=sys.stderr)

        for table_idx in valid_tables:
            counts = counts_2d[table_idx]
            analyses.append((lut_name, table_idx, counts))
            stats_row = {
                "lut": lut_name,
                "table": table_idx,
                "n_rows": int(counts.size),
                "layout": infer_row_layout(lut_name, counts.size, args.n_c, args.positional_dim),
            }
            stats_row.update(concentration_stats(counts))
            summary_rows.append(stats_row)
            add_top_rows(lut_name, table_idx, counts, args.top_n,
                         args.n_c, args.positional_dim, top_rows)
            ranges = zero_update_ranges(counts)
            zero_ranges_by_table[(lut_name, table_idx)] = ranges
            add_zero_ranges(lut_name, table_idx, counts,
                            args.n_c, args.positional_dim, zero_range_rows)
            add_bit_bias(lut_name, table_idx, counts, args.n_c,
                         args.positional_dim, bit_bias_rows)
            add_modulo_patterns(lut_name, table_idx, counts, modulo_rows)
            marginal_rows.extend(component_marginals(lut_name, table_idx, counts,
                                                     args.n_c, args.positional_dim))
            aggregate_by_lut.setdefault(lut_name, np.zeros_like(counts, dtype=np.uint64))
            aggregate_by_lut[lut_name] += counts

    if not analyses:
        print("No tables selected for analysis.", file=sys.stderr)
        return 1

    write_csv(out_dir / "table_summary.csv", summary_rows, [
        "lut", "table", "layout", "n_rows", "total_updates", "nonzero_rows",
        "max_updates", "max_share", "top3_share", "top10_share",
        "entropy", "normalized_entropy", "gini",
    ])
    write_csv(out_dir / "top_rows.csv", top_rows, [
        "lut", "table", "rank", "row0", "row1", "updates", "share", "zscore",
        "layout", "row_bits", "q_bin", "k_bin", "pe_bin", "q_bits", "k_bits",
        "pe_bits", "q_popcount", "k_popcount", "pe_popcount",
    ])
    write_csv(out_dir / "bit_bias.csv", bit_bias_rows, [
        "lut", "table", "layout", "bit", "component", "component_bit",
        "updates_with_bit_1", "share_with_bit_1",
    ])
    write_csv(out_dir / "modulo_residues.csv", modulo_rows, [
        "lut", "table", "modulo", "rank", "residue0", "residue1", "updates", "share",
    ])
    write_csv(out_dir / "component_marginals.csv", marginal_rows, [
        "lut", "table", "component", "value0", "value1", "bits", "updates", "share",
    ])
    write_csv(out_dir / "zero_update_ranges.csv", zero_range_rows, [
        "lut", "table", "rank", "start_row0", "end_row0", "start_row1", "end_row1",
        "length", "layout", "start_bits", "end_bits", "start_q_bin", "end_q_bin",
        "start_k_bin", "end_k_bin", "start_pe_bin", "end_pe_bin",
    ])

    aggregate_rows = []
    aggregate_zero_range_rows = []
    for lut_name, counts in aggregate_by_lut.items():
        add_top_rows(lut_name, "selected_sum", counts, args.top_n,
                     args.n_c, args.positional_dim, aggregate_rows)
        add_zero_ranges(lut_name, "selected_sum", counts,
                        args.n_c, args.positional_dim, aggregate_zero_range_rows)
    write_csv(out_dir / "aggregate_top_rows.csv", aggregate_rows, [
        "lut", "table", "rank", "row0", "row1", "updates", "share", "zscore",
        "layout", "row_bits", "q_bin", "k_bin", "pe_bin", "q_bits", "k_bits",
        "pe_bits", "q_popcount", "k_popcount", "pe_popcount",
    ])
    write_csv(out_dir / "aggregate_zero_update_ranges.csv", aggregate_zero_range_rows, [
        "lut", "table", "rank", "start_row0", "end_row0", "start_row1", "end_row1",
        "length", "layout", "start_bits", "end_bits", "start_q_bin", "end_q_bin",
        "start_k_bin", "end_k_bin", "start_pe_bin", "end_pe_bin",
    ])

    try:
        make_plots(out_dir, analyses, args.n_c, args.positional_dim, args.line, args.show, args.mpl_backend)
    except ImportError as exc:
        print(f"Matplotlib could not create plots: {exc}", file=sys.stderr)
        print("CSV analysis was written without plots.", file=sys.stderr)

    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"stats_npz: {stats_path}\n")
        f.write("selected_tables:\n")
        for lut_name, tables in selected.items():
            table_text = ", ".join(str(int(t)) for t in tables)
            f.write(f"  {lut_name}: {table_text}\n")
        f.write("\nMost concentrated selected tables by top3_share:\n")
        for row in sorted(summary_rows, key=lambda r: r["top3_share"], reverse=True)[:10]:
            f.write(
                f"  {row['lut']} table {row['table']}: "
                f"top3={row['top3_share']:.3f}, max={row['max_share']:.3f}, "
                f"gini={row['gini']:.3f}, nonzero={row['nonzero_rows']}/{row['n_rows']}\n"
            )
        f.write("\nZero-update row ranges in selected tables (1-based, inclusive):\n")
        for lut_name, table_idx, _counts in analyses:
            ranges = zero_ranges_by_table[(lut_name, table_idx)]
            zero_rows = sum(end0 - start0 + 1 for start0, end0 in ranges)
            f.write(
                f"  {lut_name} table {table_idx}: "
                f"{zero_rows} zero rows in {len(ranges)} ranges: {format_ranges(ranges)}\n"
            )
        f.write("\nFiles written:\n")
        for name in [
            "table_summary.csv",
            "top_rows.csv",
            "aggregate_top_rows.csv",
            "zero_update_ranges.csv",
            "aggregate_zero_update_ranges.csv",
            "component_marginals.csv",
            "bit_bias.csv",
            "modulo_residues.csv",
        ]:
            f.write(f"  {out_dir / name}\n")

    print(f"Wrote analysis to {out_dir}")
    print(f"Report: {report_path}")
    for lut_name, tables in selected.items():
        print(f"{lut_name}: tables {', '.join(str(int(t)) for t in tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
