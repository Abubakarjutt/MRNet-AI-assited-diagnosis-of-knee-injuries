import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime

from experiment_configs import DEFAULT_GROUP, EXPERIMENT_GROUPS


SUMMARY_PATTERNS = {
    "best_val_auc": re.compile(r"^best_val_auc:\s+([0-9.]+)$", re.MULTILINE),
    "best_val_loss": re.compile(r"^best_val_loss:\s+([0-9.]+)$", re.MULTILINE),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9.]+)$", re.MULTILINE),
    "avg_epoch_seconds": re.compile(r"^avg_epoch_seconds:\s+([0-9.]+)$", re.MULTILINE),
    "epochs_ran": re.compile(r"^epochs_ran:\s+([0-9]+)$", re.MULTILINE),
    "num_params_M": re.compile(r"^num_params_M:\s+([0-9.]+)$", re.MULTILINE),
    "device": re.compile(r"^device:\s+([a-z]+)$", re.MULTILINE),
}

RESULT_COLUMNS = [
    "timestamp",
    "experiment",
    "best_val_auc",
    "best_val_loss",
    "training_minutes",
    "avg_epoch_seconds",
    "epochs_ran",
    "num_params_M",
    "device",
    "status",
    "description",
    "log_file",
]


def ensure_results_file(results_path):
    if os.path.exists(results_path):
        return

    with open(results_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writeheader()


def parse_summary(log_text):
    parsed = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(log_text)
        parsed[key] = match.group(1) if match else None
    return parsed


def append_result(results_path, row):
    with open(results_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writerow(row)


def get_keep_threshold(results_path):
    best_auc = 0.0
    if not os.path.exists(results_path):
        return best_auc

    with open(results_path, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                best_auc = max(best_auc, float(row["best_val_auc"]))
            except (TypeError, ValueError):
                continue
    return best_auc


def build_command(script_dir, prefix_name, train_args):
    command = [sys.executable, "train.py", "--prefix_name", prefix_name]
    for key, value in train_args.items():
        command.extend([f"--{key}", str(value)])
    return command


def coerce_override_value(raw_value):
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return int(lowered == "true")

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def parse_overrides(entries):
    overrides = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid override '{entry}'. Expected KEY=VALUE.")
        key, value = entry.split("=", 1)
        overrides[key.strip()] = coerce_override_value(value.strip())
    return overrides


def select_experiments(group, only):
    experiments = EXPERIMENT_GROUPS[group]
    if not only:
        return experiments

    requested = set(only)
    return [experiment for experiment in experiments if experiment["name"] in requested]


def write_markdown_summary(summary_path, rows):
    if not summary_path:
        return

    with open(summary_path, "w") as handle:
        handle.write("# MRNet Experiment Summary\n\n")
        if not rows:
            handle.write("No experiments were run.\n")
            return

        handle.write("| Experiment | Status | Best Val AUC | Minutes | Device |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['experiment']} | {row['status']} | {row['best_val_auc']} "
                f"| {row['training_minutes']} | {row['device']} |\n"
            )


def run(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, args.results_file)
    logs_dir = os.path.join(script_dir, args.log_dir)
    os.makedirs(logs_dir, exist_ok=True)
    ensure_results_file(results_path)

    experiments = select_experiments(args.group, args.only)
    if not experiments:
        raise ValueError("No experiments matched the requested names.")

    overrides = parse_overrides(args.set)
    run_rows = []
    for experiment in experiments:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix_name = f"{args.run_tag}_{experiment['name']}_{timestamp}"
        log_file = os.path.join(logs_dir, f"{prefix_name}.log")
        train_args = dict(experiment["args"])
        train_args.update(overrides)
        command = build_command(script_dir, prefix_name, train_args)

        with open(log_file, "w") as handle:
            process = subprocess.run(
                command,
                cwd=script_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        with open(log_file, "r") as handle:
            log_text = handle.read()

        summary = parse_summary(log_text)
        previous_best_auc = get_keep_threshold(results_path)

        if process.returncode != 0 or summary["best_val_auc"] is None:
            status = "crash"
            best_val_auc = "0.000000"
            best_val_loss = "nan"
            training_minutes = "0.00"
            avg_epoch_seconds = "nan"
            epochs_ran = "0"
            num_params_m = "0.00"
            device = "unknown"
        else:
            best_val_auc = summary["best_val_auc"]
            best_val_loss = summary["best_val_loss"]
            training_minutes = f"{float(summary['training_seconds']) / 60.0:.2f}"
            avg_epoch_seconds = summary["avg_epoch_seconds"]
            epochs_ran = summary["epochs_ran"]
            num_params_m = summary["num_params_M"]
            device = summary["device"]
            status = "keep" if float(best_val_auc) >= previous_best_auc else "discard"

        row = {
            "timestamp": timestamp,
            "experiment": experiment["name"],
            "best_val_auc": best_val_auc,
            "best_val_loss": best_val_loss,
            "training_minutes": training_minutes,
            "avg_epoch_seconds": avg_epoch_seconds,
            "epochs_ran": epochs_ran,
            "num_params_M": num_params_m,
            "device": device,
            "status": status,
            "description": experiment["description"],
            "log_file": os.path.relpath(log_file, script_dir),
        }
        append_result(results_path, row)
        run_rows.append(row)

        print(
            f"{experiment['name']}: status={status} best_val_auc={best_val_auc} "
            f"log={os.path.relpath(log_file, script_dir)}"
        )

    write_markdown_summary(args.summary_file, run_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", type=str, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--results_file", type=str, default="results.tsv")
    parser.add_argument("--log_dir", type=str, default="experiment_logs")
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP,
        choices=sorted(EXPERIMENT_GROUPS.keys()),
        help="Experiment preset group to run.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Optional markdown summary output path for CI systems.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of experiment names to run.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=None,
        help="Optional train.py overrides in KEY=VALUE form.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
