"""Compare LUT S-table row changes across model checkpoints.

Examples:
    python compare_lut_checkpoint_rows.py results/POS/run_dir
    python compare_lut_checkpoint_rows.py model_step_0.npz model_step_10000.npz model.npz
"""

import argparse
import ast
import csv
import math
import os
import re
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PLOT_METRIC_LABELS = {
    "row-l2": "row delta L2",
    "row-rmse": "row delta RMSE",
    "row-lmin": "minimum absolute coordinate delta",
    "binary": "changed row indicator",
}


ROW_DIFF_FIELDS = [
    "pair_label", "left_label", "right_label", "left_step", "right_step",
    "left_path", "right_path", "lut", "group", "layer", "head", "component",
    "table", "dtype_left", "dtype_right", "shape", "n_tables", "total_rows",
    "y_dim", "changed_rows", "unchanged_rows", "changed_pct",
    "changed_elements", "total_elements", "changed_element_pct",
    "mean_abs_diff", "max_abs_diff",
    "plot_metric", "plot_metric_mean", "plot_metric_max",
    "plot_metric_nonzero_rows", "plot_metric_nonzero_pct",
]

SUMMARY_FIELDS = [
    "pair_label", "left_label", "right_label", "left_step", "right_step",
    "lut", "group", "layer", "head", "component", "tables_compared",
    "total_rows", "changed_rows", "unchanged_rows", "changed_pct",
    "total_elements", "changed_elements", "changed_element_pct",
    "mean_abs_diff", "max_abs_diff",
    "plot_metric", "plot_metric_mean", "plot_metric_max",
    "plot_metric_nonzero_rows", "plot_metric_nonzero_pct",
]

TIMELINE_FIELDS = [
    "pair_label", "right_label", "right_step", "lut", "group", "table",
    "total_rows", "changed_rows", "changed_pct", "plot_metric",
    "plot_metric_mean", "plot_metric_nonzero_rows", "plot_metric_nonzero_pct",
]

ROW_DELTA_TIMELINE_FIELDS = [
    "pair_label", "right_label", "right_step", "lut", "group", "table",
    "plot_metric", "plot_metric_mean", "plot_metric_nonzero_rows",
    "plot_metric_nonzero_pct",
]


@dataclass(frozen=True)
class LUTInfo:
    name: str
    key: str
    layer: int | None
    head: int | None
    component: str
    group: str


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    label: str
    step: int | None


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "lut"


def infer_lut_info(key):
    if not key.endswith("_S"):
        return None
    name = key[:-2]
    match = re.fullmatch(r"ffn_(\d+)", name)
    if match:
        return LUTInfo(name, key, int(match.group(1)), None, "ffn", "ffn")
    match = re.fullmatch(r"head_(\d+)_(\d+)_(.+)", name)
    if match:
        component = match.group(3)
        group = "new_attention" if component in {"Q", "K", "V", "Final"} else "attention"
        return LUTInfo(name, key, int(match.group(1)), int(match.group(2)), component, group)
    if name.startswith("output"):
        return LUTInfo(name, key, None, None, name, "output")
    return LUTInfo(name, key, None, None, name, "other")


def discover_luts(npz):
    luts = []
    for key in npz.files:
        info = infer_lut_info(key)
        if info is not None:
            luts.append(info)
    return sorted(luts, key=lambda item: (
        item.layer is None,
        -1 if item.layer is None else item.layer,
        -1 if item.head is None else item.head,
        item.component,
        item.name,
    ))


def parse_index_selection(text, name):
    if text is None or text.strip().lower() == "all":
        return None
    selected = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Bad {name} range {part!r}: start > end")
            if start < 0:
                raise ValueError(f"Bad {name} range {part!r}: negative indices are not supported")
            selected.update(range(start, end + 1))
        else:
            value = int(part)
            if value < 0:
                raise ValueError(f"Bad {name} value {part!r}: negative indices are not supported")
            selected.add(value)
    return selected


def parse_int_list(text, flag_name):
    if text is None:
        return None
    values = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError as exc:
            raise ValueError(
                f"Bad {flag_name} value {text!r}; expected comma-separated integers"
            ) from exc
    return values


def selected_table_indices(n_tables, table_selection):
    if table_selection is None:
        return np.arange(n_tables, dtype=np.int32)
    selected = [idx for idx in sorted(table_selection) if idx < n_tables]
    return np.array(selected, dtype=np.int32)


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


def default_label(path):
    step = step_from_path(path)
    if step is not None:
        return f"step_{step}"
    return path.stem


def step_from_path(path):
    match = re.search(r"(?:^|_)step_(\d+)(?:_|$)", path.stem)
    if match:
        return int(match.group(1))
    return None


def checkpoint_sort_key(path):
    step = step_from_path(path)
    if step is not None:
        return (0, step, path.name)
    if path.name == "model.npz":
        return (2, math.inf, path.name)
    return (1, math.inf, path.name)


def npy_header_from_zip_member(zip_file, member_name):
    with zip_file.open(member_name) as f:
        if f.read(6) != b"\x93NUMPY":
            return None
        major = f.read(1)[0]
        minor = f.read(1)[0]
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major in {2, 3}:
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            return None
        encoding = "utf-8" if major == 3 else "latin1"
        _ = minor
        return ast.literal_eval(f.read(header_len).decode(encoding).strip())


def has_model_lut(path):
    try:
        with zipfile.ZipFile(path) as zip_file:
            for member in zip_file.namelist():
                if not member.endswith(".npy"):
                    continue
                key = member[:-4]
                if infer_lut_info(key) is None:
                    continue
                header = npy_header_from_zip_member(zip_file, member)
                if header is not None and len(header.get("shape", ())) == 3:
                    return True
    except Exception:
        return False
    return False


def resolve_checkpoint_paths(inputs):
    paths = [Path(item) for item in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        candidates = sorted(paths[0].glob("*.npz"), key=checkpoint_sort_key)
        paths = [path for path in candidates if has_model_lut(path)]
    else:
        resolved = []
        for path in paths:
            candidates = [path]
            if path.suffix != ".npz":
                candidates.append(Path(str(path) + ".npz"))
            for candidate in candidates:
                if candidate.exists():
                    resolved.append(candidate)
                    break
            else:
                raise FileNotFoundError(str(path))
        paths = resolved
    if len(paths) < 2:
        raise ValueError("Need at least two model checkpoint .npz files to compare.")
    return [Checkpoint(path, default_label(path), step_from_path(path)) for path in paths]


def pairs_for_mode(checkpoints, mode):
    if mode == "first-last":
        return [(checkpoints[0], checkpoints[-1])]
    if mode == "first-all":
        return [(checkpoints[0], ckpt) for ckpt in checkpoints[1:]]
    if mode == "consecutive":
        return list(zip(checkpoints[:-1], checkpoints[1:]))
    raise ValueError(f"unknown mode {mode!r}")


def pair_label(left, right):
    return f"{left.label}_vs_{right.label}"


def lut_passes_filters(info, layers, heads, min_layer, max_layer, explicit_luts, include_other):
    if explicit_luts and info.name not in explicit_luts:
        return False
    if not explicit_luts and info.group == "other" and not include_other:
        return False
    if info.layer is None:
        return True
    if layers is not None and info.layer not in layers:
        return False
    if min_layer is not None and info.layer < min_layer:
        return False
    if max_layer is not None and info.layer > max_layer:
        return False
    if info.head is not None and heads is not None and info.head not in heads:
        return False
    return True


def row_diff_mask(left, right, atol, rtol):
    if atol == 0.0 and rtol == 0.0:
        changed_elements = left != right
    else:
        changed_elements = ~np.isclose(left, right, atol=atol, rtol=rtol)
    return np.any(changed_elements, axis=1), changed_elements


def common_record(left_ckpt, right_ckpt, info, table_idx, shape, dtype_left, dtype_right):
    return {
        "pair_label": pair_label(left_ckpt, right_ckpt),
        "left_label": left_ckpt.label,
        "right_label": right_ckpt.label,
        "left_step": "" if left_ckpt.step is None else left_ckpt.step,
        "right_step": "" if right_ckpt.step is None else right_ckpt.step,
        "left_path": str(left_ckpt.path),
        "right_path": str(right_ckpt.path),
        "lut": info.name,
        "group": info.group,
        "layer": "" if info.layer is None else info.layer,
        "head": "" if info.head is None else info.head,
        "component": info.component,
        "table": int(table_idx),
        "dtype_left": str(dtype_left),
        "dtype_right": str(dtype_right),
        "shape": "x".join(str(dim) for dim in shape),
        "n_tables": int(shape[0]),
        "total_rows": int(shape[1]),
        "y_dim": int(shape[2]),
    }


def plot_metric_values(abs_diff, row_changed, plot_metric):
    if plot_metric == "row-l2":
        return np.linalg.norm(abs_diff, axis=1)
    if plot_metric == "row-rmse":
        if abs_diff.shape[1] == 0:
            return np.zeros(abs_diff.shape[0], dtype=np.float64)
        return np.linalg.norm(abs_diff, axis=1) / math.sqrt(abs_diff.shape[1])
    if plot_metric == "row-lmin":
        if abs_diff.shape[1] == 0:
            return np.zeros(abs_diff.shape[0], dtype=np.float64)
        return np.min(abs_diff, axis=1)
    if plot_metric == "binary":
        return row_changed.astype(np.float64)
    raise ValueError(f"unknown --plot-metric {plot_metric!r}")


def compare_table(left_table, right_table, atol, rtol, plot_metric):
    row_changed, element_changed = row_diff_mask(left_table, right_table, atol, rtol)
    diff = np.asarray(right_table, dtype=np.float64) - np.asarray(left_table, dtype=np.float64)
    abs_diff = np.abs(diff)
    effective_abs_diff = np.where(element_changed, abs_diff, 0.0)
    values = plot_metric_values(effective_abs_diff, row_changed, plot_metric)
    changed_rows = int(np.count_nonzero(row_changed))
    changed_elements = int(np.count_nonzero(element_changed))
    total_rows = int(row_changed.size)
    total_elements = int(element_changed.size)
    metric_nonzero_rows = int(np.count_nonzero(values))
    return values.astype(np.float64), {
        "changed_rows": changed_rows,
        "unchanged_rows": total_rows - changed_rows,
        "changed_pct": 100.0 * changed_rows / total_rows if total_rows else 0.0,
        "changed_elements": changed_elements,
        "total_elements": total_elements,
        "changed_element_pct": 100.0 * changed_elements / total_elements if total_elements else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "plot_metric": plot_metric,
        "plot_metric_mean": float(values.mean()) if values.size else 0.0,
        "plot_metric_max": float(values.max()) if values.size else 0.0,
        "plot_metric_nonzero_rows": metric_nonzero_rows,
        "plot_metric_nonzero_pct": 100.0 * metric_nonzero_rows / total_rows if total_rows else 0.0,
    }


def compare_checkpoint_pair(
    left_ckpt, right_ckpt, args, layers, heads, table_selection, explicit_luts,
):
    detail_rows = []
    plot_counts = {}
    warnings = []
    with np.load(left_ckpt.path) as left_data, np.load(right_ckpt.path) as right_data:
        left_luts = {info.name: info for info in discover_luts(left_data)}
        right_luts = {info.name: info for info in discover_luts(right_data)}
        selected_names = []
        for name in sorted(left_luts, key=lambda n: (
            left_luts[n].layer is None,
            -1 if left_luts[n].layer is None else left_luts[n].layer,
            -1 if left_luts[n].head is None else left_luts[n].head,
            left_luts[n].component,
            n,
        )):
            info = left_luts[name]
            if lut_passes_filters(
                info, layers, heads, args.min_layer, args.max_layer,
                explicit_luts, args.include_other_luts,
            ):
                selected_names.append(name)

        missing_explicit = sorted(explicit_luts - set(left_luts))
        if missing_explicit:
            warnings.extend(f"Requested LUT {name!r} was not found in {left_ckpt.path}" for name in missing_explicit)

        for name in selected_names:
            if name not in right_luts:
                message = f"{name}: missing from {right_ckpt.path}"
                if args.allow_mismatch:
                    warnings.append(f"Skipping {message}")
                    continue
                raise ValueError(message)
            info = left_luts[name]
            left_arr = left_data[info.key]
            right_arr = right_data[right_luts[name].key]
            if left_arr.ndim != 3 or right_arr.ndim != 3:
                message = f"{name}: expected 3D arrays, got {left_arr.shape} and {right_arr.shape}"
                if args.allow_mismatch:
                    warnings.append(f"Skipping {message}")
                    continue
                raise ValueError(message)
            if left_arr.shape != right_arr.shape:
                message = f"{name}: shape mismatch {left_arr.shape} vs {right_arr.shape}"
                if args.allow_mismatch:
                    warnings.append(f"Skipping {message}")
                    continue
                raise ValueError(message)

            table_indices = selected_table_indices(left_arr.shape[0], table_selection)
            if table_indices.size == 0:
                warnings.append(f"Skipping {name}: selected tables are outside [0, {left_arr.shape[0] - 1}]")
                continue

            lut_values = np.zeros((left_arr.shape[0], left_arr.shape[1]), dtype=np.float64)
            for table_idx in table_indices:
                table_idx = int(table_idx)
                row_values, metrics = compare_table(
                    left_arr[table_idx], right_arr[table_idx],
                    args.atol, args.rtol, args.plot_metric,
                )
                lut_values[table_idx] = row_values
                row = common_record(
                    left_ckpt, right_ckpt, info, table_idx, left_arr.shape,
                    left_arr.dtype, right_arr.dtype,
                )
                row.update(metrics)
                detail_rows.append(row)
            plot_counts[info.name] = (info, lut_values, table_indices)

    return detail_rows, plot_counts, warnings


def summarize_detail_rows(detail_rows):
    grouped = {}
    for row in detail_rows:
        key = (
            row["pair_label"], row["left_label"], row["right_label"],
            row["left_step"], row["right_step"], row["lut"], row["group"],
            row["layer"], row["head"], row["component"],
        )
        item = grouped.setdefault(key, {
            "pair_label": row["pair_label"],
            "left_label": row["left_label"],
            "right_label": row["right_label"],
            "left_step": row["left_step"],
            "right_step": row["right_step"],
            "lut": row["lut"],
            "group": row["group"],
            "layer": row["layer"],
            "head": row["head"],
            "component": row["component"],
            "tables_compared": 0,
            "total_rows": 0,
            "changed_rows": 0,
            "unchanged_rows": 0,
            "total_elements": 0,
            "changed_elements": 0,
            "_weighted_abs_sum": 0.0,
            "max_abs_diff": 0.0,
            "plot_metric": row["plot_metric"],
            "_weighted_plot_metric_sum": 0.0,
            "plot_metric_max": 0.0,
            "plot_metric_nonzero_rows": 0,
        })
        item["tables_compared"] += 1
        item["total_rows"] += int(row["total_rows"])
        item["changed_rows"] += int(row["changed_rows"])
        item["unchanged_rows"] += int(row["unchanged_rows"])
        item["total_elements"] += int(row["total_elements"])
        item["changed_elements"] += int(row["changed_elements"])
        item["_weighted_abs_sum"] += float(row["mean_abs_diff"]) * int(row["total_elements"])
        item["max_abs_diff"] = max(float(item["max_abs_diff"]), float(row["max_abs_diff"]))
        item["_weighted_plot_metric_sum"] += float(row["plot_metric_mean"]) * int(row["total_rows"])
        item["plot_metric_max"] = max(float(item["plot_metric_max"]), float(row["plot_metric_max"]))
        item["plot_metric_nonzero_rows"] += int(row["plot_metric_nonzero_rows"])

    rows = []
    for item in grouped.values():
        total_rows = item["total_rows"]
        total_elements = item["total_elements"]
        item["changed_pct"] = 100.0 * item["changed_rows"] / total_rows if total_rows else 0.0
        item["changed_element_pct"] = (
            100.0 * item["changed_elements"] / total_elements if total_elements else 0.0
        )
        item["mean_abs_diff"] = item["_weighted_abs_sum"] / total_elements if total_elements else 0.0
        item["plot_metric_mean"] = item["_weighted_plot_metric_sum"] / total_rows if total_rows else 0.0
        item["plot_metric_nonzero_pct"] = (
            100.0 * item["plot_metric_nonzero_rows"] / total_rows if total_rows else 0.0
        )
        del item["_weighted_abs_sum"]
        del item["_weighted_plot_metric_sum"]
        rows.append(item)
    return sorted(rows, key=lambda row: (row["pair_label"], row["group"], row["lut"]))


def timeline_rows(detail_rows):
    return [
        {
            "pair_label": row["pair_label"],
            "right_label": row["right_label"],
            "right_step": row["right_step"],
            "lut": row["lut"],
            "group": row["group"],
            "table": row["table"],
            "total_rows": row["total_rows"],
            "changed_rows": row["changed_rows"],
            "changed_pct": row["changed_pct"],
            "plot_metric": row["plot_metric"],
            "plot_metric_mean": row["plot_metric_mean"],
            "plot_metric_nonzero_rows": row["plot_metric_nonzero_rows"],
            "plot_metric_nonzero_pct": row["plot_metric_nonzero_pct"],
        }
        for row in detail_rows
    ]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_plot_tables(info, n_tables, selected_tables, args, rng):
    requested = set(int(value) for value in selected_tables)
    sample_size = sample_size_for_group(
        info.group, args.ffn_tables, args.attention_tables, args.new_attention_tables
    )
    sampled = sample_table_indices(n_tables, sample_size, rng)
    return np.array([idx for idx in sampled if int(idx) in requested], dtype=np.int32)


def plot_metric_label(plot_metric):
    return PLOT_METRIC_LABELS[plot_metric]


def plot_metric_filename(plot_metric):
    return plot_metric.replace("-", "_")


def plot_pair_counts(plot_counts, out_dir, args, plt):
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.sample_seed)
    ylabel = plot_metric_label(args.plot_metric)
    metric_name = safe_filename(args.plot_metric)
    for lut_name in sorted(plot_counts):
        info, counts, selected_tables = plot_counts[lut_name]
        plot_tables = selected_plot_tables(info, counts.shape[0], selected_tables, args, rng)
        if plot_tables.size == 0:
            continue

        n_rows = counts.shape[1]
        rows = np.arange(1, n_rows + 1)
        if args.overlay_tables:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            for table_idx in plot_tables:
                values = counts[int(table_idx)]
                if args.line:
                    ax.plot(rows, values, linewidth=1.0, label=f"table {int(table_idx)}")
                else:
                    ax.bar(rows, values, width=1.0, alpha=0.35,
                           label=f"table {int(table_idx)}")
            ax.set_title(f"{lut_name}: {ylabel}")
            ax.set_xlabel("row")
            ax.set_ylabel(ylabel)
            ax.set_xlim(1, n_rows)
            if args.plot_metric == "binary":
                ax.set_ylim(0, 1.05)
            ax.legend(loc="best")
        else:
            fig, axes = plt.subplots(
                plot_tables.size, 1,
                figsize=(12, max(3, 2.5 * plot_tables.size)),
                squeeze=False, sharex=True,
            )
            for table_idx, ax in zip(plot_tables, axes[:, 0]):
                values = counts[int(table_idx)]
                if args.line:
                    ax.plot(rows, values, linewidth=1.0)
                else:
                    ax.bar(rows, values, width=1.0)
                nonzero = int(np.count_nonzero(values))
                pct = 100.0 * nonzero / n_rows if n_rows else 0.0
                ax.set_title(f"{lut_name} table {int(table_idx)}: {ylabel} ({pct:.2f}% non-zero)")
                ax.set_xlabel("row")
                ax.set_ylabel(ylabel)
                ax.set_xlim(1, n_rows)
                if args.plot_metric == "binary":
                    ax.set_ylim(0, 1.05)
        fig.tight_layout()
        output_path = out_dir / f"{safe_filename(lut_name)}_{metric_name}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        written.append(output_path)
        print(
            f"{lut_name}: plotted {args.plot_metric} tables "
            f"{', '.join(str(int(i)) for i in plot_tables)}"
        )
    return written


def plot_series_by_group_and_lut(out_dir, rows, metric, ylabel, title, filename_prefix, plt):
    if not rows:
        return []
    written = []
    labels = []
    for row in rows:
        label = row["right_label"]
        if label not in labels:
            labels.append(label)
    x = np.arange(len(labels))

    groups = sorted({row["group"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    for group in groups:
        y = []
        for label in labels:
            values = [row[metric] for row in rows
                      if row["right_label"] == label and row["group"] == group]
            y.append(float(np.mean(values)) if values else math.nan)
        ax.plot(x, y, marker="o", label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / f"{filename_prefix}_by_group.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    lut_names = sorted({row["lut"] for row in rows})
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(lut_names)), 5.5))
    for lut_name in lut_names:
        y = []
        for label in labels:
            values = [row[metric] for row in rows
                      if row["right_label"] == label and row["lut"] == lut_name]
            y.append(float(np.mean(values)) if values else math.nan)
        ax.plot(x, y, marker="o", linewidth=1.0, label=lut_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} by LUT")
    ax.grid(axis="y", alpha=0.25)
    if len(lut_names) <= 18:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.99, 0.02, f"{len(lut_names)} LUTs; see CSV for labels",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"{filename_prefix}_by_lut.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def plot_timeline(out_dir, rows, plt):
    written = []
    written.extend(plot_series_by_group_and_lut(
        out_dir, rows, "changed_pct", "mean rows changed (%)",
        "Changed row coverage over checkpoint comparisons",
        "changed_pct_over_steps", plt,
    ))
    if not rows:
        return written
    metric = rows[0]["plot_metric"]
    metric_file = plot_metric_filename(metric)
    ylabel = plot_metric_label(metric)
    written.extend(plot_series_by_group_and_lut(
        out_dir, rows, "plot_metric_nonzero_pct",
        f"{ylabel} non-zero rows (%)",
        f"{ylabel} non-zero coverage over checkpoint comparisons",
        f"{metric_file}_nonzero_pct_over_steps", plt,
    ))
    written.extend(plot_series_by_group_and_lut(
        out_dir, rows, "plot_metric_mean",
        f"mean {ylabel}",
        f"Mean {ylabel} over checkpoint comparisons",
        f"{metric_file}_mean_over_steps", plt,
    ))
    return written


def load_matplotlib(out_dir, show):
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def print_summary(detail_rows, checkpoints, out_dir):
    total_rows = sum(int(row["total_rows"]) for row in detail_rows)
    changed_rows = sum(int(row["changed_rows"]) for row in detail_rows)
    changed_pct = 100.0 * changed_rows / total_rows if total_rows else 0.0
    print(f"Compared {len(checkpoints)} checkpoints; wrote results to {out_dir}")
    print(f"Rows changed: {changed_rows}/{total_rows} ({changed_pct:.2f}%)")
    top = sorted(detail_rows, key=lambda row: float(row["changed_pct"]), reverse=True)[:10]
    if top:
        print("Highest-change tables:")
        for row in top:
            print(
                f"  {row['pair_label']} {row['lut']} table {row['table']}: "
                f"{row['changed_rows']}/{row['total_rows']} ({row['changed_pct']:.2f}%)"
            )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare LUT S-table row changes across saved model checkpoints."
    )
    parser.add_argument("checkpoints", nargs="+",
                        help="A checkpoint directory, or an ordered list of model .npz files")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <last checkpoint stem>_row_diffs)")
    parser.add_argument("--mode", choices=["first-last", "first-all", "consecutive"],
                        default="first-last",
                        help="Checkpoint pairs to compare (default: first-last)")
    parser.add_argument("--layers", default=None,
                        help="Layer filter, e.g. all, 0, 0-2, or 0,2")
    parser.add_argument("--heads", default=None,
                        help="Only compare these attention head indices, comma-separated")
    parser.add_argument("--min-layer", type=int, default=None,
                        help="Only compare LUTs from this layer index or later")
    parser.add_argument("--max-layer", type=int, default=None,
                        help="Only compare LUTs up to this layer index")
    parser.add_argument("--lut", action="append", default=[],
                        help="Exact LUT name to compare, e.g. ffn_0 or head_0_0_V. Can repeat.")
    parser.add_argument("--tables", default="all",
                        help="Table indices, e.g. all, 0, 0-3, or 0,2,5 (default: all)")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="Absolute tolerance for row comparison (default: 0)")
    parser.add_argument("--rtol", type=float, default=0.0,
                        help="Relative tolerance for row comparison (default: 0)")
    parser.add_argument("--plot-metric", choices=["row-l2", "row-rmse", "row-lmin", "binary"],
                        default="row-l2",
                        help="Per-row metric to plot on the Y-axis (default: row-l2)")
    parser.add_argument("--line", action="store_true",
                        help="Use line plots instead of bars")
    parser.add_argument("--overlay-tables", action="store_true",
                        help="Plot sampled tables for a LUT on one axis instead of one subplot per table")
    parser.add_argument("--ffn-tables", type=int, default=3,
                        help="Random FFN tables to plot per FFN LUT (default: 3; 0 = all)")
    parser.add_argument("--attention-tables", type=int, default=3,
                        help="Random old-attention tables to plot per LUT (default: 3; 0 = all)")
    parser.add_argument("--new-attention-tables", type=int, default=3,
                        help="Random new-attention Q/K/V/Final tables to plot per LUT (default: 3; 0 = all)")
    parser.add_argument("--include-other-luts", action="store_true",
                        help="Also compare uncategorized LUTs when --lut is not specified")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for deterministic table sampling (default: 0)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Write CSV files only")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive plot windows")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="Skip missing or shape-mismatched LUTs instead of failing")
    return parser


def main_with_args(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        if args.atol < 0.0 or args.rtol < 0.0:
            raise ValueError("--atol and --rtol must be non-negative")
        layers = parse_index_selection(args.layers, "--layers")
        heads = parse_int_list(args.heads, "--heads")
        table_selection = parse_index_selection(args.tables, "--tables")
        explicit_luts = set(args.lut or [])
        checkpoints = resolve_checkpoint_paths(args.checkpoints)
        pairs = pairs_for_mode(checkpoints, args.mode)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (
        checkpoints[-1].path.with_suffix("").parent /
        (checkpoints[-1].path.with_suffix("").name + "_row_diffs")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detail_rows = []
    all_timeline_rows = []
    all_written = []
    try:
        plt = None if args.no_plots else load_matplotlib(out_dir, args.show)
    except ImportError:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib",
              file=sys.stderr)
        return 1

    try:
        for left_ckpt, right_ckpt in pairs:
            detail_rows, plot_counts, warnings = compare_checkpoint_pair(
                left_ckpt, right_ckpt, args, layers, heads, table_selection, explicit_luts
            )
            for warning in warnings:
                print(warning, file=sys.stderr)
            if not detail_rows:
                raise ValueError(f"No LUT tables were compared for {pair_label(left_ckpt, right_ckpt)}")
            all_detail_rows.extend(detail_rows)
            all_timeline_rows.extend(timeline_rows(detail_rows))
            if plt is not None:
                pair_out_dir = out_dir / pair_label(left_ckpt, right_ckpt) if len(pairs) > 1 else out_dir
                all_written.extend(plot_pair_counts(plot_counts, pair_out_dir, args, plt))

        detail_path = out_dir / "checkpoint_row_diffs.csv"
        summary_path = out_dir / "checkpoint_row_diffs_summary.csv"
        write_csv(detail_path, all_detail_rows, ROW_DIFF_FIELDS)
        write_csv(summary_path, summarize_detail_rows(all_detail_rows), SUMMARY_FIELDS)
        all_written.extend([detail_path, summary_path])

        if args.mode in {"first-all", "consecutive"}:
            timeline_path = out_dir / "changed_pct_timeline.csv"
            write_csv(timeline_path, all_timeline_rows, TIMELINE_FIELDS)
            all_written.append(timeline_path)
            row_delta_path = out_dir / "row_delta_timeline.csv"
            write_csv(row_delta_path, all_timeline_rows, ROW_DELTA_TIMELINE_FIELDS)
            all_written.append(row_delta_path)
            if plt is not None:
                all_written.extend(plot_timeline(out_dir, all_timeline_rows, plt))

    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_summary(all_detail_rows, checkpoints, out_dir)
    for path in all_written:
        print(path)
    return 0


def main():
    return main_with_args()


if __name__ == "__main__":
    raise SystemExit(main())
