"""Compare per-table LUT update coverage stats across two runs.

The primary metric is the percentage of rows in a table that received at
least one update:

    nonzero_rows / n_rows * 100

Inputs can be either table summary CSVs or run directories. For run
directories, the script prefers hot_row_analysis_all/table_summary.csv and
falls back to lut_update_stats_summary.csv.
"""

import argparse
import csv
import math
import os
import shlex
import sys
from pathlib import Path

import numpy as np


HOT_ROW_SUMMARY = Path("hot_row_analysis_all") / "table_summary.csv"
FALLBACK_SUMMARY = Path("lut_update_stats_summary.csv")
OPTIONAL_STATS = [
    "max_updates",
    "max_share",
    "top3_share",
    "top10_share",
    "entropy",
    "normalized_entropy",
    "gini",
]
CONFIG_KEYS = [
    "num_layers",
    "num_heads",
    "n_t",
    "embedding_dim",
    "n_c",
    "positional_dim",
    "attn_qk_dim",
    "attn_qk_n_c",
    "attn_qk_n_t",
    "seed",
    "max_steps",
]


def resolve_input(path_text):
    path = Path(path_text)
    if path.is_file():
        return path, find_config_for_csv(path)
    if not path.is_dir():
        raise FileNotFoundError(path_text)

    preferred = path / HOT_ROW_SUMMARY
    fallback = path / FALLBACK_SUMMARY
    if preferred.exists():
        return preferred, path / "config.txt"
    if fallback.exists():
        return fallback, path / "config.txt"
    raise FileNotFoundError(
        f"{path} has neither {HOT_ROW_SUMMARY} nor {FALLBACK_SUMMARY}"
    )


def find_config_for_csv(csv_path):
    candidates = [
        csv_path.parent / "config.txt",
        csv_path.parent.parent / "config.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value):
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def read_rows(csv_path, label, config):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for source in reader:
            n_rows = parse_int(source.get("n_rows") or source.get("total_rows"))
            nonzero_rows = parse_int(source.get("nonzero_rows") or source.get("updated_rows"))
            if n_rows is None or nonzero_rows is None:
                continue
            total_updates = parse_int(source.get("total_updates"))
            row = {
                "run_label": label,
                "lut": source.get("lut", ""),
                "table": source.get("table", ""),
                "group": lut_group(source.get("lut", ""), config.get("attention_mode")),
                "n_rows": n_rows,
                "total_updates": total_updates,
                "nonzero_rows": nonzero_rows,
                "nonzero_pct": 100.0 * nonzero_rows / n_rows if n_rows else 0.0,
                "source_csv": str(csv_path),
                "attention_mode": config.get("attention_mode", "unknown"),
            }
            for name in OPTIONAL_STATS:
                row[name] = parse_float(source.get(name))
            if row["max_updates"] is None:
                row["max_updates"] = parse_float(source.get("max_updates_on_row"))
            rows.append(row)
    return rows


def lut_group(lut_name, attention_mode=None):
    if lut_name.startswith("ffn_"):
        return "ffn"
    if lut_name == "output" or lut_name.startswith("output"):
        return "output"
    if lut_name.startswith("head_"):
        if attention_mode == "new-attention":
            return "new_attention"
        if attention_mode == "old-attention":
            return "old_attention"
        if lut_name.endswith(("_Q", "_K", "_V", "_Final")):
            return "new_attention"
        return "old_attention"
    return "other"


def parse_config(config_path):
    config = {"attention_mode": "unknown", "config_path": ""}
    if not config_path or not Path(config_path).exists():
        return config

    config["config_path"] = str(config_path)
    command_lines = []
    in_command = False
    with open(config_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "command:":
                in_command = True
                continue
            if in_command:
                command_lines.append(line)

    command = " ".join(command_lines)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    config["attention_mode"] = "new-attention" if "--new-attention" in tokens else "old-attention"
    config.update(parse_cli_flags(tokens))
    return config


def parse_cli_flags(tokens):
    parsed = {}
    for idx, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        key = token[2:].replace("-", "_")
        if key == "new_attention":
            parsed[key] = True
            continue
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            parsed[key] = tokens[idx + 1]
        else:
            parsed[key] = True
    return parsed


def default_label(input_path, csv_path, config):
    if config.get("run_dir"):
        return Path(config["run_dir"]).name
    csv_parent = csv_path.parent
    if csv_parent.name == "hot_row_analysis_all":
        return csv_parent.parent.name
    if Path(input_path).is_dir():
        return Path(input_path).name
    return csv_path.stem


def config_caption(label, config):
    parts = [label, config.get("attention_mode", "unknown")]
    for key in CONFIG_KEYS:
        if key in config:
            parts.append(f"{key}={config[key]}")
    return ", ".join(str(part) for part in parts)


def comparison_key(row):
    lut_name = row["lut"]
    if row["group"] == "old_attention" and lut_name.endswith("_V"):
        lut_name = lut_name[:-2] + "_Final"
    return (lut_name, row["table"])


def display_lut_name(row_a, row_b):
    if row_a["lut"] == row_b["lut"]:
        return row_a["lut"]
    return f"{row_a['lut']} -> {row_b['lut']}"


def comparison_rows(rows_a, rows_b):
    by_key_a = {comparison_key(row): row for row in rows_a}
    by_key_b = {comparison_key(row): row for row in rows_b}
    matched = []
    unmatched = []

    for key in sorted(by_key_a.keys() & by_key_b.keys()):
        a = by_key_a[key]
        b = by_key_b[key]
        delta = b["nonzero_pct"] - a["nonzero_pct"]
        record = {
            "lut": display_lut_name(a, b),
            "table": key[1],
            "group_a": a["group"],
            "group_b": b["group"],
            "n_rows_a": a["n_rows"],
            "n_rows_b": b["n_rows"],
            "nonzero_rows_a": a["nonzero_rows"],
            "nonzero_rows_b": b["nonzero_rows"],
            "nonzero_pct_a": a["nonzero_pct"],
            "nonzero_pct_b": b["nonzero_pct"],
            "nonzero_pct_delta": delta,
            "nonzero_pct_relative_delta": delta / a["nonzero_pct"] if a["nonzero_pct"] else "",
            "total_updates_a": a["total_updates"] if a["total_updates"] is not None else "",
            "total_updates_b": b["total_updates"] if b["total_updates"] is not None else "",
        }
        for stat in OPTIONAL_STATS:
            record[f"{stat}_a"] = blank_if_none(a.get(stat))
            record[f"{stat}_b"] = blank_if_none(b.get(stat))
            if a.get(stat) is not None and b.get(stat) is not None:
                record[f"{stat}_delta"] = b[stat] - a[stat]
            else:
                record[f"{stat}_delta"] = ""
        matched.append(record)

    for key in sorted(by_key_a.keys() - by_key_b.keys()):
        unmatched.append(unmatched_record("a_only", by_key_a[key]))
    for key in sorted(by_key_b.keys() - by_key_a.keys()):
        unmatched.append(unmatched_record("b_only", by_key_b[key]))

    return matched, unmatched


def blank_if_none(value):
    return "" if value is None else value


def unmatched_record(side, row):
    return {
        "side": side,
        "run_label": row["run_label"],
        "lut": row["lut"],
        "table": row["table"],
        "group": row["group"],
        "n_rows": row["n_rows"],
        "nonzero_rows": row["nonzero_rows"],
        "nonzero_pct": row["nonzero_pct"],
        "source_csv": row["source_csv"],
    }


def group_summary(rows):
    summaries = []
    labels = []
    for row in rows:
        if row["run_label"] not in labels:
            labels.append(row["run_label"])
    for label in labels:
        label_rows = [row for row in rows if row["run_label"] == label]
        for group in sorted({row["group"] for row in label_rows}):
            values = np.array([row["nonzero_pct"] for row in label_rows if row["group"] == group], dtype=float)
            summaries.append({
                "run_label": label,
                "attention_mode": label_rows[0]["attention_mode"] if label_rows else "",
                "group": group,
                "count": int(values.size),
                "mean_nonzero_pct": float(values.mean()) if values.size else "",
                "median_nonzero_pct": float(np.median(values)) if values.size else "",
                "min_nonzero_pct": float(values.min()) if values.size else "",
                "max_nonzero_pct": float(values.max()) if values.size else "",
            })
    return summaries


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir, matched, unmatched, grouped):
    comparison_fields = [
        "lut", "table", "group_a", "group_b", "n_rows_a", "n_rows_b",
        "nonzero_rows_a", "nonzero_rows_b", "nonzero_pct_a", "nonzero_pct_b",
        "nonzero_pct_delta", "nonzero_pct_relative_delta",
        "total_updates_a", "total_updates_b",
    ]
    for stat in OPTIONAL_STATS:
        comparison_fields.extend([f"{stat}_a", f"{stat}_b", f"{stat}_delta"])
    write_csv(out_dir / "table_comparison.csv", matched, comparison_fields)
    write_csv(out_dir / "unmatched_tables.csv", unmatched, [
        "side", "run_label", "lut", "table", "group", "n_rows",
        "nonzero_rows", "nonzero_pct", "source_csv",
    ])
    write_csv(out_dir / "group_summary.csv", grouped, [
        "run_label", "attention_mode", "group", "count", "mean_nonzero_pct",
        "median_nonzero_pct", "min_nonzero_pct", "max_nonzero_pct",
    ])


def make_plots(out_dir, rows_a, rows_b, matched, grouped, label_a, label_b, caption_a, caption_b, show, backend):
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    import matplotlib
    if backend:
        matplotlib.use(backend, force=True)
    elif not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title_suffix = f"{caption_a}\n{caption_b}"
    plot_table_comparison(out_dir, matched, label_a, label_b, title_suffix, plt)
    plot_group_comparison(out_dir, grouped, label_a, label_b, title_suffix, plt)
    plot_distribution(out_dir, rows_a, rows_b, label_a, label_b, title_suffix, plt)
    plot_optional_stats(out_dir, matched, label_a, label_b, title_suffix, plt)

    if show:
        plt.show()
    plt.close("all")


def plot_table_comparison(out_dir, matched, label_a, label_b, title_suffix, plt):
    if not matched:
        return
    labels = [f"{row['lut']}:{row['table']}" for row in matched]
    x = np.arange(len(matched))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(10, len(matched) * 0.38), 5.5))
    ax.bar(x - width / 2, [row["nonzero_pct_a"] for row in matched], width, label=label_a)
    ax.bar(x + width / 2, [row["nonzero_pct_b"] for row in matched], width, label=label_b)
    ax.set_ylabel("non-zero update rows (% of table)")
    ax.set_title(f"Non-zero update coverage by matched table\n{title_suffix}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "nonzero_pct_by_table.png", dpi=150)
    plt.close(fig)


def plot_group_comparison(out_dir, grouped, label_a, label_b, title_suffix, plt):
    if not grouped:
        return
    groups = sorted({row["group"] for row in grouped})
    by_label_group = {(row["run_label"], row["group"]): row for row in grouped}
    x = np.arange(len(groups))
    width = 0.42
    fig, ax = plt.subplots(figsize=(9, 5))
    values_a = [value_or_nan(by_label_group.get((label_a, group), {}).get("mean_nonzero_pct")) for group in groups]
    values_b = [value_or_nan(by_label_group.get((label_b, group), {}).get("mean_nonzero_pct")) for group in groups]
    ax.bar(x - width / 2, values_a, width, label=label_a)
    ax.bar(x + width / 2, values_b, width, label=label_b)
    ax.set_ylabel("mean non-zero update rows (% of table)")
    ax.set_title(f"Mean non-zero update coverage by group\n{title_suffix}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "nonzero_pct_by_group.png", dpi=150)
    plt.close(fig)


def plot_distribution(out_dir, rows_a, rows_b, label_a, label_b, title_suffix, plt):
    values_a = [row["nonzero_pct"] for row in rows_a]
    values_b = [row["nonzero_pct"] for row in rows_b]
    if not values_a and not values_b:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 100, 31)
    if values_a:
        ax.hist(values_a, bins=bins, alpha=0.55, label=label_a)
    if values_b:
        ax.hist(values_b, bins=bins, alpha=0.55, label=label_b)
    ax.set_xlabel("non-zero update rows (% of table)")
    ax.set_ylabel("table count")
    ax.set_title(f"Distribution of non-zero update coverage\n{title_suffix}", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "nonzero_pct_distribution.png", dpi=150)
    plt.close(fig)


def plot_optional_stats(out_dir, matched, label_a, label_b, title_suffix, plt):
    stats = []
    for stat in ["gini", "top3_share", "max_share"]:
        if any(row.get(f"{stat}_a") != "" and row.get(f"{stat}_b") != "" for row in matched):
            stats.append(stat)
    if not stats:
        return

    labels = [f"{row['lut']}:{row['table']}" for row in matched]
    x = np.arange(len(matched))
    width = 0.42
    fig, axes = plt.subplots(len(stats), 1, figsize=(max(10, len(matched) * 0.38), 4 * len(stats)), squeeze=False)
    for ax, stat in zip(axes[:, 0], stats):
        vals_a = [value_or_nan(row.get(f"{stat}_a")) for row in matched]
        vals_b = [value_or_nan(row.get(f"{stat}_b")) for row in matched]
        ax.bar(x - width / 2, vals_a, width, label=label_a)
        ax.bar(x + width / 2, vals_b, width, label=label_b)
        ax.set_ylabel(stat)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Matched-table concentration stats\n{title_suffix}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "concentration_stats_by_table.png", dpi=150)
    plt.close(fig)


def value_or_nan(value):
    if value == "" or value is None:
        return math.nan
    return float(value)


def default_out_dir(path_a, path_b):
    parent_a = Path(path_a).parent if Path(path_a).is_file() else Path(path_a)
    parent_b = Path(path_b).parent if Path(path_b).is_file() else Path(path_b)
    try:
        base = Path(os.path.commonpath([parent_a.resolve(), parent_b.resolve()]))
    except ValueError:
        base = Path.cwd()
    return base / "lut_table_stats_comparison"


def main():
    parser = argparse.ArgumentParser(
        description="Compare LUT table update coverage stats across two runs or CSVs."
    )
    parser.add_argument("path_a", help="First run directory or summary CSV")
    parser.add_argument("path_b", help="Second run directory or summary CSV")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: common parent/lut_table_stats_comparison)")
    parser.add_argument("--label-a", default=None, help="Label for the first input")
    parser.add_argument("--label-b", default=None, help="Label for the second input")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plots after saving them")
    parser.add_argument("--mpl-backend", default=None,
                        help="Matplotlib backend, e.g. TkAgg or QtAgg")
    args = parser.parse_args()

    try:
        csv_a, config_path_a = resolve_input(args.path_a)
        csv_b, config_path_b = resolve_input(args.path_b)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    config_a = parse_config(config_path_a)
    config_b = parse_config(config_path_b)
    label_a = args.label_a or default_label(args.path_a, csv_a, config_a)
    label_b = args.label_b or default_label(args.path_b, csv_b, config_b)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.path_a, args.path_b)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_a = read_rows(csv_a, label_a, config_a)
    rows_b = read_rows(csv_b, label_b, config_b)
    if not rows_a or not rows_b:
        print("Both inputs must contain at least one comparable row.", file=sys.stderr)
        return 1

    matched, unmatched = comparison_rows(rows_a, rows_b)
    grouped = group_summary(rows_a + rows_b)
    write_outputs(out_dir, matched, unmatched, grouped)

    try:
        make_plots(
            out_dir, rows_a, rows_b, matched, grouped, label_a, label_b,
            config_caption(label_a, config_a), config_caption(label_b, config_b),
            args.show, args.mpl_backend,
        )
    except ImportError as exc:
        print(f"Matplotlib could not create plots: {exc}", file=sys.stderr)
        print("CSV comparison was written without plots.", file=sys.stderr)

    print(f"Input A: {csv_a}")
    print(f"Input B: {csv_b}")
    print(f"Matched tables: {len(matched)}")
    print(f"Unmatched tables: {len(unmatched)}")
    print(f"Wrote comparison to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
