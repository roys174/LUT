#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

METRICS = ("accuracy", "loss", "perplexity")


def parse_number(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def format_number(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def summarize(values):
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    count = len(finite_values)
    if count == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    mean = sum(finite_values) / count
    variance = sum((value - mean) ** 2 for value in finite_values) / count
    return {
        "count": count,
        "min": min(finite_values),
        "max": max(finite_values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def seed_from_dir(seed_dir):
    name = seed_dir.name
    if name.startswith("seed_"):
        return name[len("seed_"):]
    return name


def load_seed_rows(seed_dir):
    loss_file = seed_dir / "loss.csv"
    if not loss_file.exists():
        return []

    rows = []
    with loss_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            csv_row = {
                key.strip(): value
                for key, value in raw_row.items()
                if key is not None
            }
            step_number = parse_number(csv_row.get("step"))
            if step_number is None:
                continue
            rows.append({
                "seed": seed_from_dir(seed_dir),
                "step": int(step_number),
                "loss": parse_number(csv_row.get("loss")),
                "perplexity": parse_number(csv_row.get("perplexity")),
                "accuracy": parse_number(csv_row.get("accuracy")),
            })
    rows.sort(key=lambda row: row["step"])
    return rows


def stat_columns(prefix):
    return [
        f"{prefix}_count",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_mean",
        f"{prefix}_std",
    ]


def add_stats(output_row, prefix, values):
    stats = summarize(values)
    output_row[f"{prefix}_count"] = stats["count"]
    output_row[f"{prefix}_min"] = format_number(stats["min"])
    output_row[f"{prefix}_max"] = format_number(stats["max"])
    output_row[f"{prefix}_mean"] = format_number(stats["mean"])
    output_row[f"{prefix}_std"] = format_number(stats["std"])


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_per_seed_summary(seed_rows):
    rows = []
    for seed, metrics_rows in sorted(seed_rows.items(), key=lambda item: item[0]):
        if not metrics_rows:
            continue
        final_row = metrics_rows[-1]
        output_row = {
            "seed": seed,
            "checkpoints": len(metrics_rows),
            "first_step": metrics_rows[0]["step"],
            "last_step": final_row["step"],
            "final_accuracy": format_number(final_row["accuracy"]),
            "final_loss": format_number(final_row["loss"]),
            "final_perplexity": format_number(final_row["perplexity"]),
        }
        for metric in METRICS:
            add_stats(output_row, metric, [row[metric] for row in metrics_rows])
        rows.append(output_row)
    return rows


def build_by_step_summary(all_rows):
    rows_by_step = {}
    for row in all_rows:
        rows_by_step.setdefault(row["step"], []).append(row)

    summary_rows = []
    for step in sorted(rows_by_step):
        step_rows = rows_by_step[step]
        output_row = {
            "step": step,
            "seed_count": len({row["seed"] for row in step_rows}),
        }
        for metric in METRICS:
            add_stats(output_row, metric, [row[metric] for row in step_rows])
        summary_rows.append(output_row)
    return summary_rows


def build_final_step_summary(seed_rows, all_rows):
    if not seed_rows or not all_rows:
        return []

    step_sets = [set(row["step"] for row in rows) for rows in seed_rows.values() if rows]
    common_steps = set.intersection(*step_sets) if step_sets else set()
    if common_steps:
        final_step = max(common_steps)
        step_scope = "last_common_step"
    else:
        final_step = max(row["step"] for row in all_rows)
        step_scope = "latest_available_step"

    final_rows = [row for row in all_rows if row["step"] == final_step]
    output_row = {
        "step_scope": step_scope,
        "step": final_step,
        "seed_count": len({row["seed"] for row in final_rows}),
    }
    for metric in METRICS:
        add_stats(output_row, metric, [row[metric] for row in final_rows])
    return [output_row]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize validation metrics from a POS multi-seed run directory."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Parent run directory containing seed_*/loss.csv files.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        parser.error(f"run directory does not exist: {run_dir}")

    seed_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("seed_"))
    seed_rows = {}
    for seed_dir in seed_dirs:
        rows = load_seed_rows(seed_dir)
        if rows:
            seed_rows[seed_from_dir(seed_dir)] = rows

    all_rows = [row for rows in seed_rows.values() for row in rows]

    per_seed_fieldnames = [
        "seed",
        "checkpoints",
        "first_step",
        "last_step",
        "final_accuracy",
        "final_loss",
        "final_perplexity",
    ]
    for metric in METRICS:
        per_seed_fieldnames.extend(stat_columns(metric))

    by_step_fieldnames = ["step", "seed_count"]
    for metric in METRICS:
        by_step_fieldnames.extend(stat_columns(metric))

    final_step_fieldnames = ["step_scope", "step", "seed_count"]
    for metric in METRICS:
        final_step_fieldnames.extend(stat_columns(metric))

    write_csv(run_dir / "per_seed_summary.csv", per_seed_fieldnames, build_per_seed_summary(seed_rows))
    write_csv(run_dir / "by_step_summary.csv", by_step_fieldnames, build_by_step_summary(all_rows))
    write_csv(run_dir / "final_step_summary.csv", final_step_fieldnames, build_final_step_summary(seed_rows, all_rows))

    print(f"Wrote summaries to {run_dir}")
    print(f"Seeds summarized: {len(seed_rows)}")
    print(f"Validation rows summarized: {len(all_rows)}")


if __name__ == "__main__":
    main()
