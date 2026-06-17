"""Analyze learned LUT S tables from a saved model checkpoint.

The input is the .npz file written by main.py --save-model.  This script
compares rows inside selected tables and compares tables inside each layer LUT.

Examples:
    python analyze_lut_model_tables.py saved_models/110526_small.npz

    python analyze_lut_model_tables.py saved_models/110526_small.npz \
        --layers 0 --tables 0-3 --row-sample 1024

    python analyze_lut_model_tables.py saved_models/110526_small.npz \
        --lut ffn_0 --lut head_0_0_V --tables 0,1,2
"""

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LUTInfo:
    name: str
    key: str
    layer: int | None
    head: int | None
    component: str
    group: str


def resolve_model_path(path_text):
    path = Path(path_text)
    candidates = [path]
    if path.suffix != ".npz":
        candidates.append(Path(str(path) + ".npz"))
    candidates.append(Path("LUT/LUT") / path)
    if path.suffix != ".npz":
        candidates.append(Path("LUT/LUT") / (str(path) + ".npz"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


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


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "lut"


def infer_lut_info(key):
    if not key.endswith("_S"):
        return None

    name = key[:-2]
    match = re.fullmatch(r"ffn_(\d+)", name)
    if match:
        layer = int(match.group(1))
        return LUTInfo(name, key, layer, None, "ffn", "layer")

    match = re.fullmatch(r"head_(\d+)_(\d+)_(.+)", name)
    if match:
        layer = int(match.group(1))
        head = int(match.group(2))
        component = match.group(3)
        return LUTInfo(name, key, layer, head, component, "layer")

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


def selected_table_indices(n_tables, table_selection):
    if table_selection is None:
        return np.arange(n_tables, dtype=np.int32)
    selected = [idx for idx in sorted(table_selection) if idx < n_tables]
    return np.array(selected, dtype=np.int32)


def sample_indices(size, sample_size, rng):
    if sample_size is None or sample_size == 0 or sample_size >= size:
        return np.arange(size, dtype=np.int64)
    return np.sort(rng.choice(size, size=sample_size, replace=False).astype(np.int64))


def limited_sample_indices(size, requested_size, max_size, rng):
    requested = sample_indices(size, requested_size, rng)
    if requested.size <= max_size:
        return requested, False
    limited = np.sort(rng.choice(requested, size=max_size, replace=False).astype(np.int64))
    return limited, True


def describe_values(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_min": "",
            f"{prefix}_p05": "",
            f"{prefix}_p50": "",
            f"{prefix}_p95": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(values.max()),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def common_lut_fields(info):
    return {
        "lut": info.name,
        "layer": "" if info.layer is None else info.layer,
        "head": "" if info.head is None else info.head,
        "component": info.component,
    }


def lut_summary(info, S):
    n_tables, n_rows, y_dim = S.shape
    row_norms = np.sqrt(np.sum(S * S, axis=2))
    table_norms = np.sqrt(np.sum(S * S, axis=(1, 2)))
    row = common_lut_fields(info)
    row.update({
        "key": info.key,
        "dtype": str(S.dtype),
        "n_tables": int(n_tables),
        "n_rows": int(n_rows),
        "y_dim": int(y_dim),
        "s_min": float(S.min()),
        "s_max": float(S.max()),
        "s_mean": float(S.mean()),
        "s_std": float(S.std()),
        "row_norm_min": float(row_norms.min()),
        "row_norm_mean": float(row_norms.mean()),
        "row_norm_std": float(row_norms.std()),
        "row_norm_max": float(row_norms.max()),
        "table_norm_min": float(table_norms.min()),
        "table_norm_mean": float(table_norms.mean()),
        "table_norm_std": float(table_norms.std()),
        "table_norm_max": float(table_norms.max()),
    })
    return row


def row_pair_matrices(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    dots = X @ X.T
    denom = np.outer(norms, norms)
    cosine = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
    dist_sq = norms[:, None] ** 2 + norms[None, :] ** 2 - 2.0 * dots
    l2 = np.sqrt(np.maximum(dist_sq, 0.0))
    return cosine, l2, dots, norms


def top_indices(values, top_n, largest):
    if values.size == 0 or top_n <= 0:
        return np.array([], dtype=np.int64)
    k = min(top_n, values.size)
    if largest:
        candidate = np.argpartition(-values, k - 1)[:k]
        return candidate[np.argsort(-values[candidate])]
    candidate = np.argpartition(values, k - 1)[:k]
    return candidate[np.argsort(values[candidate])]


def add_row_pair_analysis(info, S, table_idx, row_indices, top_n, summary_rows, extreme_rows):
    table = np.asarray(S[table_idx, row_indices, :], dtype=np.float64)
    selected_rows = row_indices.size
    possible_pairs = selected_rows * (selected_rows - 1) // 2
    base = common_lut_fields(info)
    base.update({
        "table": int(table_idx),
        "rows_compared": int(selected_rows),
        "total_table_rows": int(S.shape[1]),
        "possible_pairs_compared": int(possible_pairs),
        "y_dim": int(S.shape[2]),
    })

    if selected_rows < 2:
        row = dict(base)
        row.update(describe_values([], "cosine"))
        row.update(describe_values([], "l2"))
        summary_rows.append(row)
        return

    cosine, l2, dots, norms = row_pair_matrices(table)
    upper_i, upper_j = np.triu_indices(selected_rows, k=1)
    cosine_values = cosine[upper_i, upper_j]
    l2_values = l2[upper_i, upper_j]

    row = dict(base)
    row.update(describe_values(cosine_values, "cosine"))
    row.update(describe_values(l2_values, "l2"))
    summary_rows.append(row)

    categories = [
        ("cosine_high", cosine_values, True),
        ("cosine_low", cosine_values, False),
        ("l2_nearest", l2_values, False),
        ("l2_farthest", l2_values, True),
    ]
    for category, values, largest in categories:
        for rank, pair_pos in enumerate(top_indices(values, top_n, largest), start=1):
            local_a = int(upper_i[pair_pos])
            local_b = int(upper_j[pair_pos])
            output = dict(base)
            output.update({
                "category": category,
                "rank": rank,
                "row_a": int(row_indices[local_a]),
                "row_b": int(row_indices[local_b]),
                "cosine": float(cosine[local_a, local_b]),
                "l2": float(l2[local_a, local_b]),
                "dot": float(dots[local_a, local_b]),
                "row_a_norm": float(norms[local_a]),
                "row_b_norm": float(norms[local_b]),
            })
            extreme_rows.append(output)


def compare_table_arrays(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    flat_a = A.reshape(-1)
    flat_b = B.reshape(-1)
    dot = float(np.dot(flat_a, flat_b))
    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))
    denom = norm_a * norm_b
    diff = A - B
    row_norm_a = np.linalg.norm(A, axis=1)
    row_norm_b = np.linalg.norm(B, axis=1)
    row_dot = np.einsum("ij,ij->i", A, B)
    row_denom = row_norm_a * row_norm_b
    row_cos = np.divide(row_dot, row_denom, out=np.zeros_like(row_dot), where=row_denom > 0)
    row_l2 = np.linalg.norm(diff, axis=1)
    return {
        "flat_cosine": float(dot / denom) if denom > 0 else 0.0,
        "flat_l2": float(np.linalg.norm(diff.reshape(-1))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "row_cosine_mean": float(row_cos.mean()),
        "row_cosine_std": float(row_cos.std()),
        "row_cosine_min": float(row_cos.min()),
        "row_cosine_max": float(row_cos.max()),
        "row_l2_mean": float(row_l2.mean()),
        "row_l2_std": float(row_l2.std()),
        "row_l2_min": float(row_l2.min()),
        "row_l2_max": float(row_l2.max()),
    }


def add_same_lut_table_pairs(info, S, table_indices, row_indices, rows):
    for table_a, table_b in combinations(table_indices, 2):
        metrics = compare_table_arrays(S[table_a, row_indices, :],
                                       S[table_b, row_indices, :])
        row = {
            "scope": "same_lut",
            "layer": "" if info.layer is None else info.layer,
            "lut_a": info.name,
            "table_a": int(table_a),
            "head_a": "" if info.head is None else info.head,
            "component_a": info.component,
            "lut_b": info.name,
            "table_b": int(table_b),
            "head_b": "" if info.head is None else info.head,
            "component_b": info.component,
            "rows_compared": int(row_indices.size),
            "total_table_rows": int(S.shape[1]),
            "y_dim": int(S.shape[2]),
        }
        row.update(metrics)
        rows.append(row)


def add_cross_lut_pairs(layer_items, rows):
    for left_idx, (info_a, table_a, sampled_a, total_rows_a) in enumerate(layer_items):
        for info_b, table_b, sampled_b, total_rows_b in layer_items[left_idx + 1:]:
            if info_a.name == info_b.name:
                continue
            if sampled_a.shape != sampled_b.shape:
                continue
            metrics = compare_table_arrays(sampled_a, sampled_b)
            row = {
                "scope": "same_layer_cross_lut",
                "layer": info_a.layer,
                "lut_a": info_a.name,
                "table_a": int(table_a),
                "head_a": "" if info_a.head is None else info_a.head,
                "component_a": info_a.component,
                "lut_b": info_b.name,
                "table_b": int(table_b),
                "head_b": "" if info_b.head is None else info_b.head,
                "component_b": info_b.component,
                "rows_compared": int(sampled_a.shape[0]),
                "total_table_rows": f"{total_rows_a}/{total_rows_b}",
                "y_dim": int(sampled_a.shape[1]),
            }
            row.update(metrics)
            rows.append(row)


def filter_luts(luts, layers, explicit_luts, include_output):
    selected = []
    explicit = set(explicit_luts or [])
    for info in luts:
        if explicit and info.name not in explicit:
            continue
        if not explicit:
            if info.group == "output" and not include_output:
                continue
            if info.group != "layer" and info.group != "output":
                continue
        if layers is not None and info.layer not in layers:
            continue
        selected.append(info)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Compare rows and tables in LUT model S tables saved by main.py."
    )
    parser.add_argument("model_npz", help="Path to saved model .npz from --save-model")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <model stem>_table_analysis)")
    parser.add_argument("--layers", default="all",
                        help="Layer filter, e.g. all, 0, 0-2, or 0,2 (default: all)")
    parser.add_argument("--lut", action="append", default=[],
                        help="Exact LUT name to analyze, e.g. ffn_0 or head_0_0_V. Can repeat.")
    parser.add_argument("--tables", default="all",
                        help="Table indices, e.g. all, 0, 0-3, or 0,2,5 (default: all)")
    parser.add_argument("--include-output", action="store_true",
                        help="Also analyze output/output_hi/output_lo LUTs when --lut is not specified")
    parser.add_argument("--row-sample", type=int, default=512,
                        help="Rows sampled per table for within-table row comparisons (default: 512; 0 = all)")
    parser.add_argument("--table-row-sample", type=int, default=4096,
                        help="Aligned rows sampled for table-pair comparisons (default: 4096; 0 = all)")
    parser.add_argument("--max-pairwise-rows", type=int, default=4096,
                        help="Safety cap for rows used in one row-pair matrix (default: 4096)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Extreme row pairs to write per category/table (default: 20)")
    parser.add_argument("--cross-lut", action="store_true",
                        help="Also compare same-shaped tables from different LUTs in the same layer")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for deterministic row sampling (default: 0)")
    args = parser.parse_args()

    try:
        model_path = resolve_model_path(args.model_npz)
        layers = parse_index_selection(args.layers, "--layers")
        table_selection = parse_index_selection(args.tables, "--tables")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.row_sample < 0 or args.table_row_sample < 0 or args.max_pairwise_rows < 2:
        print("Row sample sizes must be non-negative, and --max-pairwise-rows must be >= 2.",
              file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (
        model_path.with_suffix("").parent / (model_path.stem + "_table_analysis")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    row_pair_summary_rows = []
    row_pair_extreme_rows = []
    table_pair_rows = []
    cross_lut_layer_items = {}
    warnings = []

    try:
        with np.load(model_path) as data:
            all_luts = discover_luts(data)
            selected_luts = filter_luts(all_luts, layers, args.lut, args.include_output)
            missing = sorted(set(args.lut) - {info.name for info in all_luts})
            for lut_name in missing:
                warnings.append(f"Requested LUT {lut_name!r} was not found.")

            if not selected_luts:
                print("No LUT S tables matched the selection.", file=sys.stderr)
                return 1

            for info in selected_luts:
                S = data[info.key]
                if S.ndim != 3:
                    warnings.append(f"Skipping {info.name}: expected 3D S table, got {S.shape}.")
                    continue

                n_tables, n_rows, _ = S.shape
                table_indices = selected_table_indices(n_tables, table_selection)
                if table_indices.size == 0:
                    warnings.append(f"Skipping {info.name}: selected tables are outside [0, {n_tables - 1}].")
                    continue

                summary_rows.append(lut_summary(info, S))

                row_indices, limited = limited_sample_indices(
                    n_rows, args.row_sample, args.max_pairwise_rows, rng
                )
                if limited:
                    warnings.append(
                        f"{info.name}: row comparison capped at {args.max_pairwise_rows} rows."
                    )

                table_row_indices = sample_indices(n_rows, args.table_row_sample, rng)
                for table_idx in table_indices:
                    add_row_pair_analysis(
                        info, S, int(table_idx), row_indices, args.top_n,
                        row_pair_summary_rows, row_pair_extreme_rows,
                    )

                add_same_lut_table_pairs(info, S, table_indices, table_row_indices,
                                         table_pair_rows)

                if args.cross_lut and info.layer is not None:
                    layer_items = cross_lut_layer_items.setdefault(info.layer, [])
                    for table_idx in table_indices:
                        layer_items.append((
                            info,
                            int(table_idx),
                            np.asarray(S[int(table_idx), table_row_indices, :]).copy(),
                            int(n_rows),
                        ))

        if args.cross_lut:
            for layer in sorted(cross_lut_layer_items):
                add_cross_lut_pairs(cross_lut_layer_items[layer], table_pair_rows)

    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_csv(out_dir / "lut_summary.csv", summary_rows, [
        "lut", "layer", "head", "component", "key", "dtype",
        "n_tables", "n_rows", "y_dim",
        "s_min", "s_max", "s_mean", "s_std",
        "row_norm_min", "row_norm_mean", "row_norm_std", "row_norm_max",
        "table_norm_min", "table_norm_mean", "table_norm_std", "table_norm_max",
    ])
    write_csv(out_dir / "row_pair_summary.csv", row_pair_summary_rows, [
        "lut", "layer", "head", "component", "table",
        "rows_compared", "total_table_rows", "possible_pairs_compared", "y_dim",
        "cosine_mean", "cosine_std", "cosine_min", "cosine_p05", "cosine_p50",
        "cosine_p95", "cosine_max",
        "l2_mean", "l2_std", "l2_min", "l2_p05", "l2_p50", "l2_p95", "l2_max",
    ])
    write_csv(out_dir / "row_pair_extremes.csv", row_pair_extreme_rows, [
        "lut", "layer", "head", "component", "table",
        "rows_compared", "total_table_rows", "possible_pairs_compared", "y_dim",
        "category", "rank", "row_a", "row_b", "cosine", "l2", "dot",
        "row_a_norm", "row_b_norm",
    ])
    write_csv(out_dir / "table_pair_summary.csv", table_pair_rows, [
        "scope", "layer",
        "lut_a", "table_a", "head_a", "component_a",
        "lut_b", "table_b", "head_b", "component_b",
        "rows_compared", "total_table_rows", "y_dim",
        "flat_cosine", "flat_l2", "mean_abs_diff", "max_abs_diff",
        "row_cosine_mean", "row_cosine_std", "row_cosine_min", "row_cosine_max",
        "row_l2_mean", "row_l2_std", "row_l2_min", "row_l2_max",
    ])

    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"model_npz: {model_path}\n")
        f.write(f"luts_analyzed: {len(summary_rows)}\n")
        f.write(f"row_pair_tables: {len(row_pair_summary_rows)}\n")
        f.write(f"table_pairs: {len(table_pair_rows)}\n")
        f.write(f"row_sample: {args.row_sample}\n")
        f.write(f"table_row_sample: {args.table_row_sample}\n")
        f.write(f"cross_lut: {args.cross_lut}\n")
        if warnings:
            f.write("warnings:\n")
            for warning in warnings:
                f.write(f"- {warning}\n")

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Wrote LUT model table analysis to {out_dir}")
    print(f"- {out_dir / 'lut_summary.csv'}")
    print(f"- {out_dir / 'row_pair_summary.csv'}")
    print(f"- {out_dir / 'row_pair_extremes.csv'}")
    print(f"- {out_dir / 'table_pair_summary.csv'}")
    print(f"- {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
