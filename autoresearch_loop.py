import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime


SUMMARY_PATTERNS = {
    "best_val_auc": re.compile(r"^best_val_auc:\s+([0-9.]+)$", re.MULTILINE),
    "best_val_loss": re.compile(r"^best_val_loss:\s+([0-9.]+)$", re.MULTILINE),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9.]+)$", re.MULTILINE),
    "epochs_ran": re.compile(r"^epochs_ran:\s+([0-9]+)$", re.MULTILINE),
    "num_params_M": re.compile(r"^num_params_M:\s+([0-9.]+)$", re.MULTILINE),
    "device": re.compile(r"^device:\s+([a-z]+)$", re.MULTILINE),
}

RESULT_COLUMNS = [
    "timestamp",
    "iteration",
    "candidate_name",
    "parent_name",
    "status",
    "best_val_auc",
    "best_val_loss",
    "training_seconds",
    "epochs_ran",
    "num_params_M",
    "device",
    "mutations",
    "config_path",
    "log_file",
]

SEARCH_SPACE = {
    "model_type": ["resnet18", "mobilenet_v3_small", "efficientnet_b0"],
    "pooling": ["max", "mean", "lse"],
    "projection_dim": [0, 128, 192, 256, 384, 512],
    "hidden_dim": [128, 192, 256, 384, 512, 768],
    "fusion_depth": [1, 2, 3],
    "dropout": [0.1, 0.15, 0.2, 0.25, 0.3],
    "lr": [1e-4, 2e-4, 3e-4, 4e-4, 6e-4],
    "weight_decay": [0.0, 1e-5, 1e-4, 5e-4],
    "image_size": [160, 192, 224],
    "cache_size": [16, 32, 48],
    "num_workers": [2, 4],
}

DEFAULT_CONFIG = {
    "model_type": "resnet18",
    "pooling": "max",
    "projection_dim": 0,
    "hidden_dim": 256,
    "fusion_depth": 2,
    "dropout": 0.2,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "image_size": 224,
    "cache_size": 32,
    "num_workers": 2,
    "epochs": 8,
    "patience": 3,
    "amp": 1,
    "channels_last": 1,
    "time_budget_minutes": 5,
    "pretrained": 0,
    "mmap": 1,
    "save_model": 0,
    "batch_size": 1,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path, default):
    if not os.path.exists(path):
        return deepcopy(default)
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def ensure_results_file(path):
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writeheader()


def append_result(path, row):
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writerow(row)


def parse_summary(log_text):
    parsed = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(log_text)
        parsed[key] = match.group(1) if match else None
    return parsed


def float_or_default(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_adjacent(value, choices, rng):
    ordered = list(choices)
    index = ordered.index(value)
    candidate_indexes = [index]
    if index > 0:
        candidate_indexes.append(index - 1)
    if index < len(ordered) - 1:
        candidate_indexes.append(index + 1)
    return ordered[rng.choice(candidate_indexes)]


def mutate_config(base_config, rng):
    candidate = deepcopy(base_config)
    mutation_count = rng.randint(1, 3)
    mutation_keys = rng.sample(list(SEARCH_SPACE.keys()), mutation_count)
    mutations = []

    for key in mutation_keys:
        old_value = candidate[key]
        new_value = choose_adjacent(old_value, SEARCH_SPACE[key], rng)
        if new_value == old_value and len(SEARCH_SPACE[key]) > 1:
            alternatives = [choice for choice in SEARCH_SPACE[key] if choice != old_value]
            new_value = rng.choice(alternatives)
        candidate[key] = new_value
        mutations.append(f"{key}:{old_value}->{new_value}")

    if candidate["fusion_depth"] == 1:
        candidate["hidden_dim"] = base_config["hidden_dim"]

    return candidate, mutations


def build_train_command(prefix_name, config_path, data_root):
    return [
        sys.executable,
        "train.py",
        "--prefix_name",
        prefix_name,
        "--data_root",
        data_root,
        "--search_config",
        config_path,
    ]


def run_candidate(script_dir, logs_dir, config_dir, cache_dir, iteration, candidate, data_root):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_name = f"iter{iteration:03d}_{timestamp}"
    config_path = os.path.join(config_dir, f"{candidate_name}.json")
    log_path = os.path.join(logs_dir, f"{candidate_name}.log")
    save_json(config_path, candidate)

    command = build_train_command(candidate_name, config_path, data_root)
    env = os.environ.copy()
    env["TORCH_HOME"] = cache_dir
    with open(log_path, "w") as handle:
        process = subprocess.run(
            command,
            cwd=script_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    with open(log_path, "r") as handle:
        log_text = handle.read()

    summary = parse_summary(log_text)
    return candidate_name, config_path, log_path, process.returncode, summary, log_text


def should_retry_without_pretrained(returncode, log_text, candidate_config):
    if returncode == 0 or int(candidate_config.get("pretrained", 0)) != 1:
        return False

    retry_markers = [
        "CERTIFICATE_VERIFY_FAILED",
        "urllib.error.URLError",
        "ssl.SSLCertVerificationError",
        "download.pytorch.org/models",
    ]
    return any(marker in log_text for marker in retry_markers)


def write_markdown_summary(path, state, rows):
    with open(path, "w") as handle:
        handle.write("# MRNet Autoresearch Summary\n\n")
        best = state["best"]
        handle.write(f"Best candidate: `{best['name']}`\n\n")
        handle.write(f"Best validation AUC: `{best['best_val_auc']:.6f}`\n\n")
        handle.write("## Current Best Config\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(best["config"], indent=2, sort_keys=True))
        handle.write("\n```\n\n")
        handle.write("## Latest Iterations\n\n")
        handle.write("| Iteration | Candidate | Status | Val AUC | Mutations |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['iteration']} | {row['candidate_name']} | {row['status']} | "
                f"{row['best_val_auc']} | {row['mutations']} |\n"
            )


def apply_runtime_overrides(config, args):
    updated = deepcopy(config)
    if args.time_budget_minutes is not None:
        updated["time_budget_minutes"] = args.time_budget_minutes
    if args.max_train_batches is not None:
        updated["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        updated["max_val_batches"] = args.max_val_batches
    if args.pretrained is not None:
        updated["pretrained"] = args.pretrained
    if args.num_workers is not None:
        updated["num_workers"] = args.num_workers
    return updated


def run(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    state_dir = os.path.expanduser(args.state_dir)
    ensure_dir(state_dir)
    logs_dir = os.path.join(state_dir, "logs")
    config_dir = os.path.join(state_dir, "configs")
    cache_dir = os.path.join(state_dir, "torch_cache")
    ensure_dir(logs_dir)
    ensure_dir(config_dir)
    ensure_dir(cache_dir)

    results_path = os.path.join(state_dir, "results.tsv")
    state_path = os.path.join(state_dir, "state.json")
    best_config_path = os.path.join(state_dir, "best_config.json")
    ensure_results_file(results_path)

    state = load_json(
        state_path,
        {
            "iteration": 0,
            "best": {
                "name": "seed",
                "best_val_auc": 0.0,
                "best_val_loss": None,
                "config": deepcopy(DEFAULT_CONFIG),
            },
        },
    )
    save_json(best_config_path, state["best"]["config"])

    rng = random.Random(args.seed + int(state["iteration"]))
    run_rows = []

    for _ in range(args.iterations):
        state["iteration"] += 1
        parent = deepcopy(state["best"])
        candidate_config, mutations = mutate_config(parent["config"], rng)
        candidate_config = apply_runtime_overrides(candidate_config, args)
        candidate_name, config_path, log_path, returncode, summary, log_text = run_candidate(
            script_dir=script_dir,
            logs_dir=logs_dir,
            config_dir=config_dir,
            cache_dir=cache_dir,
            iteration=state["iteration"],
            candidate=candidate_config,
            data_root=args.data_root,
        )
        if should_retry_without_pretrained(returncode, log_text, candidate_config):
            candidate_config["pretrained"] = 0
            mutations.append("pretrained:1->0(auto-retry)")
            candidate_name, config_path, log_path, returncode, summary, log_text = run_candidate(
                script_dir=script_dir,
                logs_dir=logs_dir,
                config_dir=config_dir,
                cache_dir=cache_dir,
                iteration=state["iteration"],
                candidate=candidate_config,
                data_root=args.data_root,
            )

        candidate_auc = float_or_default(summary["best_val_auc"])
        status = "crash"
        if returncode == 0 and summary["best_val_auc"] is not None:
            status = "keep" if candidate_auc > parent["best_val_auc"] else "discard"

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": state["iteration"],
            "candidate_name": candidate_name,
            "parent_name": parent["name"],
            "status": status,
            "best_val_auc": f"{candidate_auc:.6f}",
            "best_val_loss": summary["best_val_loss"] or "nan",
            "training_seconds": summary["training_seconds"] or "0",
            "epochs_ran": summary["epochs_ran"] or "0",
            "num_params_M": summary["num_params_M"] or "0",
            "device": summary["device"] or "unknown",
            "mutations": ", ".join(mutations),
            "config_path": config_path,
            "log_file": log_path,
        }
        append_result(results_path, row)
        run_rows.append(row)

        if status == "keep":
            state["best"] = {
                "name": candidate_name,
                "best_val_auc": candidate_auc,
                "best_val_loss": float_or_default(summary["best_val_loss"], None),
                "config": candidate_config,
            }
            save_json(best_config_path, candidate_config)

        save_json(state_path, state)
        print(
            f"iteration={state['iteration']} candidate={candidate_name} "
            f"status={status} val_auc={candidate_auc:.6f}"
        )

    if args.summary_file:
        write_markdown_summary(args.summary_file, state, run_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--state_dir",
        type=str,
        default="~/.mrnet_autoresearch",
        help="Persistent directory for autoresearch state so scheduled runs keep improving.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="MRNet-v1.0",
        help="Dataset root passed through to train.py.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Optional markdown summary output path for CI systems.",
    )
    parser.add_argument("--time_budget_minutes", type=float, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--pretrained", type=int, choices=[0, 1], default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
