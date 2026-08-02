"""Plot LUT update counts and aggregated j statistics produced by main.py.

Usage:
    python plot_lut_update_histograms.py lut_update_stats.npz

The positional input is the raw row-update .npz written by --lut-update-stats-file.
If a companion <stem>_j_stats.npz exists, this script also plots aggregate j-bin
usage, component marginals, and qkpe/final_v joint distributions.
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


COMPONENT_ORDER = {"q": 0, "k": 1, "pe": 2, "qkpe": 3, "final_v": 4, "j": 5}


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


def default_j_stats_path(stats_path):
    return stats_path.with_name(stats_path.stem + "_j_stats.npz")


def default_coord_stats_path(stats_path):
    return stats_path.with_name(stats_path.stem + "_coord_stats.npz")


def load_coord_stats(path):
    coord_path = Path(path)
    if not coord_path.exists():
        return None, {}
    if coord_path.suffix != ".npz":
        raise ValueError("coordinate statistics must be loaded from a .npz file")
    with np.load(coord_path) as data:
        return coord_path, {name: data[name].copy() for name in data.files}


def load_j_stats(path):
    j_path = Path(path)
    if not j_path.exists():
        return None, {"simple": {}, "components": {}, "joints": {}}
    if j_path.suffix != ".npz":
        raise ValueError("j statistics must be loaded from a .npz file")

    parsed = {"simple": {}, "components": defaultdict(dict), "joints": {}}
    with np.load(j_path) as data:
        for key in data.files:
            arr = data[key].copy()
            if key.startswith("simple__"):
                parsed["simple"][key[len("simple__"):]] = arr
            elif key.startswith("component__"):
                parts = key.split("__")
                if len(parts) != 3:
                    print(f"Skipping malformed j-stat key {key!r}", file=sys.stderr)
                    continue
                _, lut_name, component = parts
                parsed["components"][lut_name][component] = arr
            elif key.startswith("joint__"):
                parts = key.split("__")
                if len(parts) != 4:
                    print(f"Skipping malformed j-stat key {key!r}", file=sys.stderr)
                    continue
                _, lut_name, row_component, col_component = parts
                parsed["joints"][(lut_name, row_component, col_component)] = arr
            else:
                print(f"Skipping unknown j-stat key {key!r}", file=sys.stderr)
    parsed["components"] = dict(parsed["components"])
    return j_path, parsed


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


def plot_j_distribution(ax, counts, title, use_line=False):
    bins = np.arange(counts.shape[0])
    if use_line:
        ax.plot(bins, counts, linewidth=1.0)
    else:
        ax.bar(bins, counts, width=1.0)
    ax.set_title(title)
    ax.set_xlabel("j bin")
    ax.set_ylabel("observations")
    if counts.shape[0] > 1:
        ax.set_xlim(-0.5, counts.shape[0] - 0.5)


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


def sample_size_for_group(group, ffn_sample, attention_sample, new_attention_sample):
    if group == "ffn":
        return ffn_sample
    if group == "attention":
        return attention_sample
    if group == "new_attention":
        return new_attention_sample
    return None


def selected_tables_for_name(name, n_tables, rng, ffn_sample, attention_sample,
                             new_attention_sample, include_other):
    group = lut_group(name)
    if group == "other" and not include_other:
        return np.array([], dtype=np.int32)
    return sample_table_indices(
        n_tables,
        sample_size_for_group(group, ffn_sample, attention_sample, new_attention_sample),
        rng,
    )


def distribution_records(kind, lut_name, component, table_idx, counts, top_n):
    total = int(counts.sum())
    if counts.size == 0:
        return []
    rows = []
    order = np.argsort(counts)[::-1][:min(top_n, counts.size)]
    for rank, bin_idx in enumerate(order, start=1):
        count = int(counts[bin_idx])
        if count == 0:
            break
        rows.append({
            "record_type": "top",
            "kind": kind,
            "lut": lut_name,
            "component": component,
            "table": int(table_idx),
            "rank": rank,
            "bin": int(bin_idx),
            "bin2": "",
            "count": count,
            "share": float(count / total) if total else 0.0,
        })
    for rank, bin_idx in enumerate(np.flatnonzero(counts == 0)[:top_n], start=1):
        rows.append({
            "record_type": "zero",
            "kind": kind,
            "lut": lut_name,
            "component": component,
            "table": int(table_idx),
            "rank": rank,
            "bin": int(bin_idx),
            "bin2": "",
            "count": 0,
            "share": 0.0,
        })
    nonzero = np.flatnonzero(counts > 0)
    if nonzero.size:
        scarce = nonzero[np.argsort(counts[nonzero])[:top_n]]
        for rank, bin_idx in enumerate(scarce, start=1):
            count = int(counts[bin_idx])
            rows.append({
                "record_type": "scarce_nonzero",
                "kind": kind,
                "lut": lut_name,
                "component": component,
                "table": int(table_idx),
                "rank": rank,
                "bin": int(bin_idx),
                "bin2": "",
                "count": count,
                "share": float(count / total) if total else 0.0,
            })
    return rows


def joint_records(lut_name, row_component, col_component, table_idx, counts, top_n):
    flat = counts.ravel()
    total = int(flat.sum())
    if flat.size == 0:
        return []
    component = f"{row_component}_x_{col_component}"
    rows = []
    order = np.argsort(flat)[::-1][:min(top_n, flat.size)]
    for rank, flat_idx in enumerate(order, start=1):
        count = int(flat[flat_idx])
        if count == 0:
            break
        row_idx, col_idx = np.unravel_index(int(flat_idx), counts.shape)
        rows.append({
            "record_type": "top",
            "kind": "joint",
            "lut": lut_name,
            "component": component,
            "table": int(table_idx),
            "rank": rank,
            "bin": int(row_idx),
            "bin2": int(col_idx),
            "count": count,
            "share": float(count / total) if total else 0.0,
        })
    for rank, flat_idx in enumerate(np.flatnonzero(flat == 0)[:top_n], start=1):
        row_idx, col_idx = np.unravel_index(int(flat_idx), counts.shape)
        rows.append({
            "record_type": "zero",
            "kind": "joint",
            "lut": lut_name,
            "component": component,
            "table": int(table_idx),
            "rank": rank,
            "bin": int(row_idx),
            "bin2": int(col_idx),
            "count": 0,
            "share": 0.0,
        })
    return rows


def write_j_report(out_dir, j_stats, selected_by_name, top_n):
    rows = []
    for lut_name, counts in sorted(j_stats["simple"].items()):
        for table_idx in selected_by_name.get(lut_name, []):
            rows.extend(distribution_records(
                "simple", lut_name, "j", table_idx, counts[table_idx], top_n))
    for lut_name, components in sorted(j_stats["components"].items()):
        for component, counts in sorted(components.items(), key=lambda item: COMPONENT_ORDER.get(item[0], 99)):
            for table_idx in selected_by_name.get(lut_name, []):
                rows.extend(distribution_records(
                    "component", lut_name, component, table_idx, counts[table_idx], top_n))
    for (lut_name, row_component, col_component), counts in sorted(j_stats["joints"].items()):
        for table_idx in selected_by_name.get(lut_name, []):
            rows.extend(joint_records(
                lut_name, row_component, col_component, table_idx, counts[table_idx], top_n))

    if not rows:
        return None
    report_path = out_dir / "j_stats_top.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "record_type", "kind", "lut", "component", "table", "rank",
            "bin", "bin2", "count", "share",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def plot_row_stats(stats, out_dir, args, plt, rng, filters, samples):
    layer_filter, head_filter = filters
    ffn_sample, attention_sample, new_attention_sample = samples
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
        selected_tables = selected_tables_for_name(
            lut_name, n_tables, rng, ffn_sample, attention_sample,
            new_attention_sample, args.include_other_luts)

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
        print(f"{lut_name}: plotted row tables {', '.join(str(int(i)) for i in selected_tables)}")
        if args.show:
            plt.show()
        plt.close(fig)
    return written


def selected_j_tables(j_stats, args, rng, filters, samples):
    layer_filter, head_filter = filters
    ffn_sample, attention_sample, new_attention_sample = samples
    selected = {}
    names = set(j_stats["simple"])
    names.update(j_stats["components"])
    names.update(name for name, _, _ in j_stats["joints"])
    for lut_name in sorted(names):
        if not lut_passes_filters(lut_name, layer_filter, head_filter,
                                  args.min_layer, args.max_layer):
            continue
        arr = j_stats["simple"].get(lut_name)
        if arr is None and lut_name in j_stats["components"]:
            arr = next(iter(j_stats["components"][lut_name].values()))
        if arr is None:
            for (joint_name, _, _), joint in j_stats["joints"].items():
                if joint_name == lut_name:
                    arr = joint
                    break
        if arr is None:
            continue
        tables = selected_tables_for_name(
            lut_name, arr.shape[0], rng, ffn_sample, attention_sample,
            new_attention_sample, args.include_other_luts)
        if tables.size:
            selected[lut_name] = tables
    return selected


def plot_j_stats(j_stats, out_dir, args, plt, selected_by_name):
    written = []
    j_dir = out_dir / "j_stats"
    j_dir.mkdir(parents=True, exist_ok=True)

    for lut_name, counts in sorted(j_stats["simple"].items()):
        selected_tables = selected_by_name.get(lut_name, np.array([], dtype=np.int32))
        if selected_tables.size == 0:
            continue
        fig, axes = plt.subplots(selected_tables.size, 1,
                                 figsize=(12, max(3, 2.5 * selected_tables.size)),
                                 squeeze=False, sharex=True)
        for table_idx, ax in zip(selected_tables, axes[:, 0]):
            plot_j_distribution(ax, counts[table_idx],
                                f"{lut_name} table {table_idx}: j observations",
                                use_line=args.line)
        fig.tight_layout()
        output_path = j_dir / f"simple_{_safe_filename(lut_name)}.png"
        fig.savefig(output_path, dpi=150)
        written.append(output_path)
        print(f"{lut_name}: plotted simple j tables {', '.join(str(int(i)) for i in selected_tables)}")
        if args.show:
            plt.show()
        plt.close(fig)

    for lut_name, components in sorted(j_stats["components"].items()):
        selected_tables = selected_by_name.get(lut_name, np.array([], dtype=np.int32))
        if selected_tables.size == 0:
            continue
        ordered = sorted(components.items(), key=lambda item: COMPONENT_ORDER.get(item[0], 99))
        for table_idx in selected_tables:
            fig, axes = plt.subplots(len(ordered), 1,
                                     figsize=(12, max(3, 2.4 * len(ordered))),
                                     squeeze=False)
            for (component, counts), ax in zip(ordered, axes[:, 0]):
                plot_j_distribution(ax, counts[table_idx],
                                    f"{lut_name} table {table_idx}: {component} j observations",
                                    use_line=args.line)
            fig.tight_layout()
            output_path = j_dir / f"components_{_safe_filename(lut_name)}_table_{int(table_idx)}.png"
            fig.savefig(output_path, dpi=150)
            written.append(output_path)
            if args.show:
                plt.show()
            plt.close(fig)
        print(f"{lut_name}: plotted component j tables {', '.join(str(int(i)) for i in selected_tables)}")

    for (lut_name, row_component, col_component), counts in sorted(j_stats["joints"].items()):
        selected_tables = selected_by_name.get(lut_name, np.array([], dtype=np.int32))
        if selected_tables.size == 0:
            continue
        for table_idx in selected_tables:
            table_counts = counts[table_idx]
            row_marginal = table_counts.sum(axis=1)
            col_marginal = table_counts.sum(axis=0)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            im = axes[0].imshow(table_counts, cmap="viridis", aspect="auto", origin="lower")
            axes[0].set_title(f"{row_component} x {col_component}")
            axes[0].set_xlabel(f"{col_component} j bin")
            axes[0].set_ylabel(f"{row_component} j bin")
            fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
            plot_j_distribution(axes[1], row_marginal, f"{row_component} marginal", use_line=args.line)
            plot_j_distribution(axes[2], col_marginal, f"{col_component} marginal", use_line=args.line)
            fig.suptitle(f"{lut_name} table {table_idx}: joint j observations")
            fig.tight_layout()
            output_path = j_dir / f"joint_{_safe_filename(lut_name)}_{row_component}_x_{col_component}_table_{int(table_idx)}.png"
            fig.savefig(output_path, dpi=150)
            written.append(output_path)
            if args.show:
                plt.show()
            plt.close(fig)
        print(f"{lut_name}: plotted joint j tables {', '.join(str(int(i)) for i in selected_tables)}")
    return written


STEP_RE = re.compile(r"^(?P<stem>.+)_step_(?P<step>\d+)\.npz$")


def snapshot_step(path):
    match = STEP_RE.match(Path(path).name)
    return int(match.group("step")) if match else None


def snapshot_label(path):
    step = snapshot_step(path)
    return f"step_{step}" if step is not None else Path(path).stem


def snapshot_sort_key(path):
    step = snapshot_step(path)
    return (step is None, step if step is not None else 10**18, Path(path).name)


def discover_stats_inputs(input_path, out_dir_arg):
    path = Path(input_path)
    if path.is_dir() or (not path.exists() and path.suffix == ""):
        out_dir = Path(out_dir_arg) if out_dir_arg else path
        run_dir = path.parent
        snapshots = sorted(run_dir.glob("lut_update_stats_step_*.npz"), key=snapshot_sort_key)
        stats_files = [p for p in snapshots if not p.name.endswith(("_j_stats.npz", "_coord_stats.npz"))]
        if not stats_files:
            base = run_dir / "lut_update_stats.npz"
            if base.exists():
                stats_files = [base]
        if not stats_files:
            raise FileNotFoundError(f"No lut_update_stats*.npz files found next to {path}")
        return stats_files, out_dir, True

    stats_path = path
    out_dir = Path(out_dir_arg) if out_dir_arg else stats_path.with_suffix("").parent / (stats_path.stem + "_row_plots")
    return [stats_path], out_dir, False


def selected_tables_for_stats(stats, args, rng, filters, samples):
    layer_filter, head_filter = filters
    ffn_sample, attention_sample, new_attention_sample = samples
    selected = {}
    for lut_name in sorted(stats):
        if not lut_passes_filters(lut_name, layer_filter, head_filter,
                                  args.min_layer, args.max_layer):
            continue
        counts = stats[lut_name]
        if counts.ndim != 2:
            continue
        tables = selected_tables_for_name(
            lut_name, counts.shape[0], rng, ffn_sample, attention_sample,
            new_attention_sample, args.include_other_luts)
        if tables.size:
            selected[lut_name] = tables
    return selected


def coverage_rows_for_stats(stats, step_label, args, rng, filters, samples):
    rows = []
    selected = selected_tables_for_stats(stats, args, rng, filters, samples)
    for lut_name, tables in selected.items():
        counts = stats[lut_name]
        group = lut_group(lut_name)
        for table_idx in tables:
            row_counts = counts[int(table_idx)]
            n_rows = int(row_counts.size)
            nonzero_rows = int(np.count_nonzero(row_counts))
            rows.append({
                "step_label": step_label,
                "step": snapshot_step_from_label(step_label),
                "lut": lut_name,
                "group": group,
                "table": int(table_idx),
                "n_rows": n_rows,
                "nonzero_rows": nonzero_rows,
                "nonzero_pct": 100.0 * nonzero_rows / n_rows if n_rows else 0.0,
                "total_updates": int(row_counts.sum()),
            })
    return rows


def snapshot_step_from_label(label):
    if label.startswith("step_"):
        try:
            return int(label.split("_", 1)[1])
        except ValueError:
            return ""
    return ""


def write_coverage_timeline(out_dir, coverage_rows):
    if not coverage_rows:
        return None
    path = out_dir / "nonzero_pct_timeline.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step_label", "step", "lut", "group", "table", "n_rows",
            "nonzero_rows", "nonzero_pct", "total_updates",
        ])
        writer.writeheader()
        writer.writerows(coverage_rows)
    return path


def plot_coverage_timeline(out_dir, coverage_rows, plt):
    if not coverage_rows:
        return []
    written = []
    labels = []
    for row in coverage_rows:
        if row["step_label"] not in labels:
            labels.append(row["step_label"])
    x = np.arange(len(labels))

    groups = sorted({row["group"] for row in coverage_rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    for group in groups:
        y = []
        for label in labels:
            values = [row["nonzero_pct"] for row in coverage_rows
                      if row["step_label"] == label and row["group"] == group]
            y.append(float(np.mean(values)) if values else math.nan)
        ax.plot(x, y, marker="o", label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean non-zero update rows (% of table)")
    ax.set_title("Non-zero update coverage over saved snapshots")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "nonzero_pct_over_steps_by_group.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    lut_names = sorted({row["lut"] for row in coverage_rows})
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(lut_names)), 5.5))
    for lut_name in lut_names:
        y = []
        for label in labels:
            values = [row["nonzero_pct"] for row in coverage_rows
                      if row["step_label"] == label and row["lut"] == lut_name]
            y.append(float(np.mean(values)) if values else math.nan)
        ax.plot(x, y, marker="o", linewidth=1.0, label=lut_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean non-zero update rows (% of sampled tables)")
    ax.set_title("Non-zero update coverage over saved snapshots by LUT")
    ax.grid(axis="y", alpha=0.25)
    if len(lut_names) <= 18:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.99, 0.02, f"{len(lut_names)} LUTs; see CSV for labels",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "nonzero_pct_over_steps_by_lut.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def selected_tables_for_coord_stats(coord_stats, args, rng, filters, samples):
    layer_filter, head_filter = filters
    ffn_sample, attention_sample, new_attention_sample = samples
    selected = {}
    for lut_name in sorted(coord_stats):
        if not lut_passes_filters(lut_name, layer_filter, head_filter,
                                  args.min_layer, args.max_layer):
            continue
        counts = coord_stats[lut_name]
        if counts.ndim != 2:
            print(f"Skipping coord stats {lut_name}: expected 2D array, got shape {counts.shape}",
                  file=sys.stderr)
            continue
        tables = selected_tables_for_name(
            lut_name, counts.shape[0], rng, ffn_sample, attention_sample,
            new_attention_sample, args.include_other_luts)
        if tables.size:
            selected[lut_name] = tables
    return selected


def coord_summary_text(counts):
    total = int(counts.sum())
    updated = int(np.count_nonzero(counts))
    zero = int(counts.size - updated)
    return (
        f"mean={float(counts.mean()):.2f}\n"
        f"var={float(counts.var()):.2f}\n"
        f"zero={zero}/{counts.size}\n"
        f"total={total}\n"
        f"max={int(counts.max()) if counts.size else 0}"
    )


def plot_coord_stats(coord_stats, out_dir, args, plt, selected_by_name):
    written = []
    coord_dir = out_dir / "coord_stats"
    coord_dir.mkdir(parents=True, exist_ok=True)
    for lut_name, counts in sorted(coord_stats.items()):
        selected_tables = selected_by_name.get(lut_name, np.array([], dtype=np.int32))
        if selected_tables.size == 0:
            continue
        fig, axes = plt.subplots(selected_tables.size, 1,
                                 figsize=(12, max(3, 2.7 * selected_tables.size)),
                                 squeeze=False, sharex=True)
        for table_idx, ax in zip(selected_tables, axes[:, 0]):
            coord_counts = counts[int(table_idx)]
            coords = np.arange(coord_counts.shape[0])
            if args.line:
                ax.plot(coords, coord_counts, linewidth=1.0)
            else:
                ax.bar(coords, coord_counts, width=1.0)
            ax.set_title(f"{lut_name} table {int(table_idx)}: coordinate update counts")
            ax.set_xlabel("coordinate")
            ax.set_ylabel("non-zero update events")
            if coord_counts.shape[0] > 1:
                ax.set_xlim(-0.5, coord_counts.shape[0] - 0.5)
            ax.text(0.99, 0.95, coord_summary_text(coord_counts),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
        fig.tight_layout()
        output_path = coord_dir / f"coords_{_safe_filename(lut_name)}.png"
        fig.savefig(output_path, dpi=150)
        written.append(output_path)
        print(f"{lut_name}: plotted coordinate tables {', '.join(str(int(i)) for i in selected_tables)}")
        if args.show:
            plt.show()
        plt.close(fig)
    return written


def coord_timeline_rows_for_stats(coord_stats, step_label, args, rng, filters, samples):
    rows = []
    selected = selected_tables_for_coord_stats(coord_stats, args, rng, filters, samples)
    for lut_name, tables in selected.items():
        counts = coord_stats[lut_name]
        group = lut_group(lut_name)
        for table_idx in tables:
            coord_counts = counts[int(table_idx)]
            total_coords = int(coord_counts.size)
            updated = int(np.count_nonzero(coord_counts))
            rows.append({
                "step_label": step_label,
                "step": snapshot_step_from_label(step_label),
                "lut": lut_name,
                "group": group,
                "table": int(table_idx),
                "total_coordinate_updates": int(coord_counts.sum()),
                "updated_coordinates": updated,
                "total_coordinates": total_coords,
                "zero_updated_coordinates": total_coords - updated,
                "coord_nonzero_pct": 100.0 * updated / total_coords if total_coords else 0.0,
                "mean_updates_per_coordinate": float(coord_counts.mean()) if total_coords else 0.0,
                "var_updates_per_coordinate": float(coord_counts.var()) if total_coords else 0.0,
                "max_updates_on_coordinate": int(coord_counts.max()) if total_coords else 0,
            })
    return rows


def write_coord_timeline(out_dir, rows):
    if not rows:
        return None
    path = out_dir / "coord_stats_timeline.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step_label", "step", "lut", "group", "table",
            "total_coordinate_updates", "updated_coordinates", "total_coordinates",
            "zero_updated_coordinates", "coord_nonzero_pct",
            "mean_updates_per_coordinate", "var_updates_per_coordinate",
            "max_updates_on_coordinate",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_coord_timeline(out_dir, rows, plt):
    if not rows:
        return []
    written = []
    labels = []
    for row in rows:
        if row["step_label"] not in labels:
            labels.append(row["step_label"])
    groups = sorted({row["group"] for row in rows})
    x = np.arange(len(labels))

    for metric, ylabel, filename in [
        ("coord_nonzero_pct", "coordinates updated at least once (%)", "coord_nonzero_pct_over_steps_by_group.png"),
        ("mean_updates_per_coordinate", "mean non-zero update events per coordinate", "coord_mean_updates_over_steps_by_group.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for group in groups:
            y = []
            for label in labels:
                values = [row[metric] for row in rows
                          if row["step_label"] == label and row["group"] == group]
                y.append(float(np.mean(values)) if values else math.nan)
            ax.plot(x, y, marker="o", label=group)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Coordinate update statistics over saved snapshots")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Plot LUT row index vs updates, plus aggregate j statistics when present."
    )
    parser.add_argument("stats_npz", help="Path to raw LUT update stats .npz, or a lut_update_plots directory to discover snapshots")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for PNG plots (default: <npz stem>_row_plots, or the input plot directory)")
    parser.add_argument("--j-stats-npz", default=None,
                        help="Path to companion j-stats .npz for single-file mode (default: <stats stem>_j_stats.npz)")
    parser.add_argument("--no-j-stats", action="store_true",
                        help="Only plot row-update stats; do not load or plot companion j stats")
    parser.add_argument("--coord-stats-npz", default=None,
                        help="Path to companion coordinate-stats .npz for single-file mode (default: <stats stem>_coord_stats.npz)")
    parser.add_argument("--no-coord-stats", action="store_true",
                        help="Do not load or plot companion coordinate stats")
    parser.add_argument("--j-top-n", type=int, default=20,
                        help="Top/zero/scarce j bins to write to j_stats_top.csv (default: 20)")
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

    try:
        stats_paths, out_dir, discovered_from_plot_dir = discover_stats_inputs(args.stats_npz, args.out_dir)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

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

    ffn_sample = None if args.ffn_tables == 0 else args.ffn_tables
    attention_sample = None if args.attention_tables == 0 else args.attention_tables
    new_attention_sample = None if args.new_attention_tables == 0 else args.new_attention_tables
    samples = (ffn_sample, attention_sample, new_attention_sample)
    try:
        layer_filter = parse_int_list(args.layers, "--layers") if args.layers else None
        head_filter = parse_int_list(args.heads, "--heads") if args.heads else None
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    filters = (layer_filter, head_filter)

    all_written = []
    coverage_rows = []
    coord_timeline_rows = []
    for stats_path in stats_paths:
        try:
            stats = load_stats(stats_path)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not stats:
            print(f"No arrays found in {stats_path}", file=sys.stderr)
            return 1

        label = snapshot_label(stats_path)
        step_out_dir = out_dir / label if len(stats_paths) > 1 else out_dir
        step_out_dir.mkdir(parents=True, exist_ok=True)

        row_rng = np.random.default_rng(args.sample_seed)
        all_written.extend(plot_row_stats(stats, step_out_dir, args, plt, row_rng, filters, samples))
        timeline_rng = np.random.default_rng(args.sample_seed)
        coverage_rows.extend(coverage_rows_for_stats(stats, label, args, timeline_rng, filters, samples))

        coord_path = None
        coord_stats = {}
        if not args.no_coord_stats:
            try:
                requested_coord_path = Path(args.coord_stats_npz) if args.coord_stats_npz and len(stats_paths) == 1 else default_coord_stats_path(stats_path)
                coord_path, coord_stats = load_coord_stats(requested_coord_path)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        if coord_path is not None:
            coord_rng = np.random.default_rng(args.sample_seed)
            selected_coord_tables = selected_tables_for_coord_stats(coord_stats, args, coord_rng, filters, samples)
            all_written.extend(plot_coord_stats(coord_stats, step_out_dir, args, plt, selected_coord_tables))
            timeline_coord_rng = np.random.default_rng(args.sample_seed)
            coord_timeline_rows.extend(coord_timeline_rows_for_stats(
                coord_stats, label, args, timeline_coord_rng, filters, samples))
            print(f"Loaded coordinate stats from {coord_path}")
        elif not args.no_coord_stats:
            print(f"No coordinate stats found at {args.coord_stats_npz or default_coord_stats_path(stats_path)}")

        j_path = None
        j_stats = {"simple": {}, "components": {}, "joints": {}}
        if not args.no_j_stats:
            try:
                requested_j_path = Path(args.j_stats_npz) if args.j_stats_npz and len(stats_paths) == 1 else default_j_stats_path(stats_path)
                j_path, j_stats = load_j_stats(requested_j_path)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        if j_path is not None:
            j_rng = np.random.default_rng(args.sample_seed)
            selected_by_name = selected_j_tables(j_stats, args, j_rng, filters, samples)
            report_path = write_j_report(step_out_dir, j_stats, selected_by_name, args.j_top_n)
            all_written.extend(plot_j_stats(j_stats, step_out_dir, args, plt, selected_by_name))
            print(f"Loaded j stats from {j_path}")
            if report_path is not None:
                all_written.append(report_path)
                print(f"Wrote j-stat top/zero/scarce report to {report_path}")
        elif not args.no_j_stats:
            print(f"No j stats found at {args.j_stats_npz or default_j_stats_path(stats_path)}")

    if len(stats_paths) > 1:
        timeline_csv = write_coverage_timeline(out_dir, coverage_rows)
        if timeline_csv is not None:
            all_written.append(timeline_csv)
        all_written.extend(plot_coverage_timeline(out_dir, coverage_rows, plt))
        coord_timeline_csv = write_coord_timeline(out_dir, coord_timeline_rows)
        if coord_timeline_csv is not None:
            all_written.append(coord_timeline_csv)
        all_written.extend(plot_coord_timeline(out_dir, coord_timeline_rows, plt))

    for path in all_written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
