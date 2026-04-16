import argparse
import contextlib
import csv
import fcntl
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from glob import glob
from copy import deepcopy
from datetime import datetime

from research_priors import RESEARCH_PRIORS
from research_controller import propose_next_experiment, write_research_memo


SUMMARY_PATTERNS = {
    "best_val_auc": re.compile(r"^best_val_auc:\s+([0-9.]+)$", re.MULTILINE),
    "best_val_loss": re.compile(r"^best_val_loss:\s+([0-9.]+)$", re.MULTILINE),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9.]+)$", re.MULTILINE),
    "epochs_ran": re.compile(r"^epochs_ran:\s+([0-9]+)$", re.MULTILINE),
    "num_params_M": re.compile(r"^num_params_M:\s+([0-9.]+)$", re.MULTILINE),
    "device": re.compile(r"^device:\s+([a-z]+)$", re.MULTILINE),
"model_complexity": re.compile(r"^model_complexity:\s+([0-9.]+)$", re.MULTILINE),
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
     "model_complexity",
    "mutations",
    "config_path",
    "log_file",
    "research_memo",
]

SEARCH_SPACE = {
    "model_type": ["resnet18", "mobilenet_v3_small", "efficientnet_b0"],
    "pooling": ["max", "mean", "lse", "attention", "gem"],
    "projection_dim": [0, 128, 192, 256, 384, 512],
    "hidden_dim": [128, 192, 256, 384, 512, 768],
    "fusion_depth": [1, 2, 3],
    "fusion_gate": ["none", "se"],
    "dropout": [0.1, 0.15, 0.2, 0.25, 0.3],
    "lr": [1e-4, 2e-4, 3e-4, 4e-4, 6e-4],
    "weight_decay": [0.0, 1e-5, 1e-4, 5e-4],
    "image_size": [160, 192, 224],
    "cache_size": [16, 32, 48],
    "num_workers": [2, 4],
    "aug_policy": ["none", "light", "strong", "knee_mri"],
    "aug_noise_std": [0.0, 0.01, 0.02, 0.04],
    "aug_cutout_frac": [0.0, 0.08, 0.12, 0.18],
    "aug_slice_dropout": [0.0, 0.03, 0.06, 0.1],
}

DEFAULT_CONFIG = {
    "model_type": "resnet18",
    "pooling": "max",
    "projection_dim": 0,
    "hidden_dim": 256,
    "fusion_depth": 2,
    "fusion_gate": "none",
    "dropout": 0.2,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "image_size": 224,
    "cache_size": 32,
    "num_workers": 2,
    "aug_policy": "none",
    "aug_noise_std": 0.0,
    "aug_cutout_frac": 0.0,
    "aug_slice_dropout": 0.0,
    "epochs": 8,
    "patience": 3,
    "amp": 1,
    "channels_last": 1,
    "time_budget_minutes": 60,
    "pretrained": 0,
    "mmap": 1,
    "save_model": 0,
    "batch_size": 1,
}

LOCK_FILENAME = "loop.lock"
TIMEOUT_GRACE_MINUTES = 15
BEST_MODEL_FILENAME = "best_model.pth"
BEST_MODEL_META_FILENAME = "best_model_meta.json"
NON_PERSISTED_OVERRIDE_KEYS = {
    "max_train_batches",
    "max_val_batches",
    "save_model",
}
AUC_IMPROVEMENT_EPS = 1e-6
AUC_TIE_EPS = 1e-6
SIMPLICITY_PARAM_RATIO = 0.9
MAX_DUPLICATE_RETRIES = 16

MODEL_COMPLEXITY = {
    "mobilenet_v3_small": 0.0,
    "resnet18": 0.2,
    "efficientnet_b0": 0.4,
}

POOLING_COMPLEXITY = {
    "max": 0.0,
    "mean": 0.0,
    "lse": 0.2,
    "gem": 0.3,
    "attention": 0.5,
}


class LockUnavailableError(RuntimeError):
    pass


class LowDiskSpaceError(RuntimeError):
    pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def format_gb(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def ensure_free_space(path, min_free_gb, context):
    if min_free_gb is None or float(min_free_gb) <= 0:
        return

    free_bytes = shutil.disk_usage(path).free
    min_free_bytes = int(float(min_free_gb) * (1024 ** 3))
    if free_bytes < min_free_bytes:
        raise LowDiskSpaceError(
            f"Stopped early because free space at {path} is {format_gb(free_bytes)}, "
            f"below the required {float(min_free_gb):.1f} GiB while {context}."
        )


def load_json(path, default):
    if not os.path.exists(path):
        return deepcopy(default)
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return deepcopy(default)


def merge_missing_defaults(config, defaults):
    merged = deepcopy(defaults)
    merged.update(config)
    return merged


def sanitize_search_config(config):
    sanitized = merge_missing_defaults(config, DEFAULT_CONFIG)
    for key, choices in SEARCH_SPACE.items():
        if sanitized.get(key) not in choices:
            sanitized[key] = deepcopy(DEFAULT_CONFIG[key])
    return sanitized


def config_complexity_score(config):
    return (
        MODEL_COMPLEXITY.get(config.get("model_type"), 0.5)
        + POOLING_COMPLEXITY.get(config.get("pooling"), 0.3)
        + 0.2 * max(int(config.get("fusion_depth", 1)) - 1, 0)
        + 0.15 * int(config.get("fusion_gate") == "se")
        + float(config.get("projection_dim", 0)) / 5120.0
        + float(config.get("hidden_dim", 256)) / 5120.0
        + float(config.get("dropout", 0.2))
    )


def merge_research_state_defaults(research_state):
    merged = initialize_research_state()
    for name, value in (research_state or {}).items():
        if name in merged and isinstance(value, dict):
            merged[name].update(value)
    return merged


def atomic_write_text(path, text):
    directory = os.path.dirname(path) or "."
    ensure_dir(directory)
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def save_json(path, data):
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def normalize_result_row(row):
    return {column: row.get(column, "") for column in RESULT_COLUMNS}


def read_results_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="	")
        return [normalize_result_row(row) for row in reader]


def find_result_row(rows, candidate_name):
    for row in reversed(rows):
        if row.get("candidate_name") == candidate_name:
            return row
    return None


def serialize_results(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RESULT_COLUMNS, delimiter="	", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(normalize_result_row(row) for row in rows)
    return buffer.getvalue()


def config_signature(config):
    normalized = persistent_config_snapshot(sanitize_search_config(config))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def ensure_results_file(path):
    if not os.path.exists(path):
        atomic_write_text(path, serialize_results([]))
        return

    rows = read_results_rows(path)
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle, delimiter="	")
        current_header = next(reader, [])
    if current_header != RESULT_COLUMNS:
        atomic_write_text(path, serialize_results(rows))


def append_result(path, row):
    rows = read_results_rows(path)
    rows.append(row)
    atomic_write_text(path, serialize_results(rows))


@contextlib.contextmanager
def advisory_lock(lock_path):
    ensure_dir(os.path.dirname(lock_path) or ".")
    with open(lock_path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockUnavailableError(f"another autoresearch run is already using {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
        if key not in candidate:
            candidate[key] = DEFAULT_CONFIG[key]
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


def initialize_research_state():
    return {
        prior["name"]: {
            "trials": 0,
            "keeps": 0,
            "source": prior["source"],
        }
        for prior in RESEARCH_PRIORS
    }


def select_research_prior(research_state, rng):
    # Prioritize baseline and simpler models when trials are low
    # This encourages exploration of simple architectures first
    scored = []
    for prior in RESEARCH_PRIORS:
        stats = research_state.get(prior["name"], {"trials": 0, "keeps": 0})
        trial_penalty = stats["trials"]
        success_bonus = stats["keeps"] * 0.25

        # Bonus for simple models when we haven't explored much yet
        simplicity_bonus = 0.0
        if stats["trials"] < 3:
            # Give priority to baseline and simple models early on
            if prior["name"] in ["simple_mean_pool_linear", "fast_mobilenet_screen", "gem_slice_pooling"]:
                simplicity_bonus = 0.5
            elif prior["name"] in ["mil_attention_pooling", "se_fusion_gate", "compact_highres_backbone"]:
                simplicity_bonus = 0.3

        # Penalize complex models until simpler ones are explored
        complexity_penalty = 0.0
        if stats["trials"] < 2:
            complex_priors = [
                "transmil_multi_instance", "cross_attention_fusion", "swin_transformer_lite",
                "mae_pretrain_refinement", "adaptive_depth_network", "dinov2_efficient_backbone"
            ]
            if prior["name"] in complex_priors:
                complexity_penalty = 0.3

        novelty_bonus = rng.random() * 0.05
        score = -trial_penalty + success_bonus + simplicity_bonus - complexity_penalty + novelty_bonus
        scored.append((score, prior))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def apply_research_prior(base_config, prior, rng):
    candidate = deepcopy(base_config)
    mutations = [f"research_prior:{prior['name']}"]
    for key, value in prior["mutations"].items():
        old_value = candidate.get(key)
        candidate[key] = value
        if old_value != value:
            mutations.append(f"{key}:{old_value}->{value}")

    if rng.random() < 0.5:
        candidate, local_mutations = mutate_config(candidate, rng)
        mutations.extend(f"local:{item}" for item in local_mutations)

    return candidate, mutations


def build_train_command(prefix_name, config_path, data_root, init_checkpoint=None):
    command = [
        sys.executable,
        "train.py",
        "--prefix_name",
        prefix_name,
        "--data_root",
        data_root,
        "--search_config",
        config_path,
    ]
    if init_checkpoint:
        command.extend(["--init_checkpoint", init_checkpoint])
    return command


def model_dir(storage_root):
    return os.path.join(storage_root, "models")


def best_model_path(storage_root):
    return os.path.join(model_dir(storage_root), BEST_MODEL_FILENAME)


def best_model_meta_path(storage_root):
    return os.path.join(model_dir(storage_root), BEST_MODEL_META_FILENAME)


def candidate_model_paths(storage_root, candidate_name):
    pattern = os.path.join(model_dir(storage_root), f"model_{candidate_name}_*.pth")
    return sorted(glob(pattern))


def remove_paths(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue


def prune_model_dir(storage_root, keep_paths=None):
    keep_paths = {os.path.abspath(path) for path in (keep_paths or [])}
    model_root = model_dir(storage_root)
    if not os.path.isdir(model_root):
        return
    for path in glob(os.path.join(model_root, "*.pth")):
        if os.path.abspath(path) not in keep_paths:
            remove_paths([path])


def load_best_model_meta(storage_root):
    return load_json(best_model_meta_path(storage_root), default={})


def save_best_model_meta(storage_root, metadata):
    os.makedirs(model_dir(storage_root), exist_ok=True)
    save_json(best_model_meta_path(storage_root), metadata)


def best_checkpoint_for_parent(storage_root, parent):
    checkpoint_path = best_model_path(storage_root)
    meta = load_best_model_meta(storage_root)
    if not os.path.isfile(checkpoint_path):
        return None

    if not meta:
        return None

    if meta.get("candidate_name") != parent.get("name"):
        return None

    if meta.get("config_signature") != config_signature(parent.get("config", {})):
        return None

    return checkpoint_path


def promote_candidate_model(storage_root, candidate_name, candidate_config, candidate_auc):
    paths = candidate_model_paths(storage_root, candidate_name)
    if not paths:
        return None

    os.makedirs(model_dir(storage_root), exist_ok=True)
    promoted_src = max(paths, key=os.path.getmtime)
    promoted_dst = best_model_path(storage_root)
    if os.path.abspath(promoted_src) != os.path.abspath(promoted_dst):
        if os.path.exists(promoted_dst):
            os.remove(promoted_dst)
        os.replace(promoted_src, promoted_dst)

    prune_model_dir(storage_root, keep_paths=[promoted_dst])
    save_best_model_meta(
        storage_root,
        {
            "candidate_name": candidate_name,
            "config_signature": config_signature(candidate_config),
            "best_val_auc": candidate_auc,
        },
    )
    return promoted_dst


def discard_candidate_model(storage_root, candidate_name):
    remove_paths(candidate_model_paths(storage_root, candidate_name))
    keep_path = best_model_path(storage_root)
    keep_paths = [keep_path] if os.path.isfile(keep_path) else []
    prune_model_dir(storage_root, keep_paths=keep_paths)


def run_candidate(
    script_dir,
    logs_dir,
    config_dir,
    cache_dir,
    temp_root_dir,
    iteration,
    candidate,
    data_root,
    model_storage_root,
    init_checkpoint=None,
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_name = f"iter{iteration:03d}_{timestamp}"
    config_path = os.path.join(config_dir, f"{candidate_name}.json")
    log_path = os.path.join(logs_dir, f"{candidate_name}.log")
    execution_candidate = deepcopy(candidate)
    execution_candidate["save_model"] = 1
    save_json(config_path, execution_candidate)

    command = build_train_command(candidate_name, config_path, data_root, init_checkpoint=init_checkpoint)
    env = os.environ.copy()
    env["TORCH_HOME"] = cache_dir
    env["MRNET_MODEL_DIR"] = model_dir(model_storage_root)
    candidate_tmp_dir = os.path.join(temp_root_dir, candidate_name)
    ensure_dir(candidate_tmp_dir)
    env["TMPDIR"] = candidate_tmp_dir
    env["TMP"] = candidate_tmp_dir
    env["TEMP"] = candidate_tmp_dir
    timeout_seconds = None
    time_budget_minutes = candidate.get("time_budget_minutes")
    if time_budget_minutes is not None:
        timeout_seconds = max(
            60,
            int((float(time_budget_minutes) + TIMEOUT_GRACE_MINUTES) * 60),
        )

    returncode = 1
    timed_out = False
    with open(log_path, "w") as handle:
        try:
            process = subprocess.run(
                command,
                cwd=script_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            handle.write(
                "\n[autoresearch] Candidate timed out after "
                f"{timeout_seconds} seconds and was terminated.\n"
            )
            handle.flush()

    with open(log_path, "r") as handle:
        log_text = handle.read()

    if timed_out:
        log_text += (
            "\n[autoresearch] Candidate timed out after "
            f"{timeout_seconds} seconds and was terminated.\n"
        )

    shutil.rmtree(candidate_tmp_dir, ignore_errors=True)

    summary = parse_summary(log_text)
    return candidate_name, config_path, log_path, returncode, summary, log_text


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


def write_markdown_summary(path, state, rows, notes=None):
    with open(path, "w") as handle:
        handle.write("# MRNet Autoresearch Summary\n\n")
        best = state["best"]
        handle.write(f"Best candidate: `{best['name']}`\n\n")
        handle.write(f"Best validation AUC: `{best['best_val_auc']:.6f}`\n\n")
        if notes:
            handle.write("## Notes\n\n")
            for note in notes:
                handle.write(f"- {note}\n")
            handle.write("\n")
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


def persistent_config_snapshot(config):
    snapshot = deepcopy(config)
    for key in NON_PERSISTED_OVERRIDE_KEYS:
        snapshot.pop(key, None)
    return snapshot


def load_seen_signatures(config_dir):
    signatures = set()
    if not os.path.isdir(config_dir):
        return signatures

    for name in os.listdir(config_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(config_dir, name)
        try:
            with open(path, "r") as handle:
                signatures.add(config_signature(json.load(handle)))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return signatures


def select_next_candidate(parent, args, rng, seen_signatures, force_baseline):
    if force_baseline:
        candidate_config, mutations, prior = make_baseline_candidate(args)
        return candidate_config, mutations, prior

    last_candidate = None
    last_mutations = None
    last_prior = None
    for _ in range(MAX_DUPLICATE_RETRIES):
        prior = select_research_prior(parent["research"], rng)
        candidate_config, mutations = apply_research_prior(parent["config"], prior, rng)
        candidate_config = sanitize_search_config(candidate_config)
        candidate_config = apply_runtime_overrides(candidate_config, args)
        if config_signature(candidate_config) not in seen_signatures:
            return candidate_config, mutations, prior
        last_candidate = candidate_config
        last_mutations = list(mutations) + ["duplicate_config_retry"]
        last_prior = prior

    return last_candidate, last_mutations or ["duplicate_config_retry"], last_prior or {
        "name": "duplicate_config_retry"
    }


def is_fresh_state(state, rows):
    if rows:
        return False
    return int(state.get("iteration", 0)) == 0 and state.get("best", {}).get("name") == "seed"


def make_baseline_candidate(args):
    candidate = sanitize_search_config(DEFAULT_CONFIG)
    candidate = apply_runtime_overrides(candidate, args)
    return candidate, ["baseline"], {"name": "baseline"}


def backfill_best_metadata(state, result_rows):
    best = state.get("best", {})
    best.setdefault("num_params_M", None)
    best.setdefault("complexity_score", config_complexity_score(best.get("config", {})))
    if best.get("num_params_M") is None and best.get("name") not in {None, "seed"}:
        row = find_result_row(result_rows, best["name"])
        if row is not None:
            best["num_params_M"] = float_or_default(row.get("num_params_M"), None)
             # Use model_complexity from training output when available
            if row.get("model_complexity") is not None:
                best["complexity_score"] = float_or_default(row["model_complexity"], best.get("complexity_score"))
    return state


def make_best_entry(name, best_val_auc, best_val_loss, num_params_M, config):
    persistent_config = persistent_config_snapshot(sanitize_search_config(config))
    return {
        "name": name,
        "best_val_auc": float_or_default(best_val_auc, 0.0),
        "best_val_loss": float_or_default(best_val_loss, None),
        "num_params_M": float_or_default(num_params_M, None),
        "complexity_score": config_complexity_score(persistent_config),
        "config": persistent_config,
    }


def load_result_config(row):
    config_path = row.get("config_path")
    if not config_path:
        return None
    try:
        with open(config_path, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def is_simpler_candidate(parent, candidate_config, candidate_num_params, candidate_complexity=None):
    parent_num_params = parent.get("num_params_M")
    if (
        parent_num_params is not None
        and candidate_num_params is not None
        and candidate_num_params > 0
        and candidate_num_params <= parent_num_params * SIMPLICITY_PARAM_RATIO
    ):
        return True

    parent_complexity = parent.get(
        "complexity_score",
        config_complexity_score(parent.get("config", {})),
    )
    if candidate_complexity is None:
        candidate_complexity = config_complexity_score(candidate_config)
    return candidate_complexity < parent_complexity


def is_better_objective_candidate(parent, candidate_auc, candidate_config, candidate_num_params, candidate_complexity=None):
    parent_auc = float_or_default(parent.get("best_val_auc"), 0.0)
    if candidate_auc > parent_auc + AUC_IMPROVEMENT_EPS:
        return True

    if abs(candidate_auc - parent_auc) <= AUC_TIE_EPS and is_simpler_candidate(
        parent, candidate_config, candidate_num_params, candidate_complexity
    ):
        return True

    return False


def candidate_status(parent, candidate_auc, candidate_config, summary, returncode):
    if returncode != 0 or summary["best_val_auc"] is None:
        return "crash"

    if parent["name"] == "seed":
        return "keep"

    candidate_num_params = float_or_default(summary["num_params_M"], None)
    candidate_complexity = float_or_default(summary.get("model_complexity"), None)
    if is_better_objective_candidate(parent, candidate_auc, candidate_config, candidate_num_params, candidate_complexity):
        return "keep"

    return "discard"


def recover_best_from_history(state, result_rows):
    best = state.get("best", {})
    recovered_best = None

    if best.get("name") not in {None, "seed"}:
        recovered_best = make_best_entry(
            name=best.get("name", "seed"),
            best_val_auc=best.get("best_val_auc", 0.0),
            best_val_loss=best.get("best_val_loss"),
            num_params_M=best.get("num_params_M"),
            config=best.get("config", {}),
        )

    for row in result_rows:
        if row.get("status") == "crash":
            continue
        if row.get("best_val_auc") in {None, "", "nan"}:
            continue
        config = load_result_config(row)
        if config is None:
            continue

        candidate = make_best_entry(
            name=row.get("candidate_name", "unknown"),
            best_val_auc=row.get("best_val_auc", 0.0),
            best_val_loss=row.get("best_val_loss"),
            num_params_M=row.get("num_params_M"),
            config=config,
        )
        if recovered_best is None or is_better_objective_candidate(
            recovered_best,
            candidate["best_val_auc"],
            candidate["config"],
            candidate.get("num_params_M"),
        ):
            recovered_best = candidate

    if recovered_best is not None:
        state["best"] = recovered_best

    return state


def write_lock_summary(path, message):
    if not path:
        return
    with open(path, "w") as handle:
        handle.write("# MRNet Autoresearch Summary\n\n")
        handle.write(f"{message}\n")


def write_latest_run_manifest(state_dir, run_rows):
    manifest_path = os.path.join(state_dir, "latest_run_files.json")
    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": [
            {
                "iteration": row["iteration"],
                "candidate_name": row["candidate_name"],
                "status": row["status"],
                "log_file": row["log_file"],
                "config_path": row["config_path"],
            }
            for row in run_rows
        ],
    }
    save_json(manifest_path, manifest)


def write_best_only_manifest(state_dir, best_name):
    manifest_path = os.path.join(state_dir, "latest_run_files.json")
    row = {
        "iteration": None,
        "candidate_name": best_name,
        "status": "best",
        "log_file": os.path.join(state_dir, "logs", f"{best_name}.log"),
        "config_path": os.path.join(state_dir, "configs", f"{best_name}.json"),
    }
    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": [row],
    }
    save_json(manifest_path, manifest)


def controller_memo_path(research_dir, candidate_name):
    return os.path.join(research_dir, f"{candidate_name}.json")


def summarize_outcome(status, candidate_name, candidate_auc, summary, config_path, log_path):
    return {
        "candidate_name": candidate_name,
        "status": status,
        "best_val_auc": candidate_auc,
        "best_val_loss": summary.get("best_val_loss"),
        "training_seconds": summary.get("training_seconds"),
        "epochs_ran": summary.get("epochs_ran"),
        "num_params_M": summary.get("num_params_M"),
        "model_complexity": summary.get("model_complexity"),
        "config_path": config_path,
        "log_path": log_path,
    }


def run(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    state_dir = os.path.expanduser(args.state_dir)
    ensure_dir(state_dir)
    logs_dir = os.path.join(state_dir, "logs")
    config_dir = os.path.join(state_dir, "configs")
    cache_dir = os.path.join(state_dir, "torch_cache")
    temp_root_dir = os.path.join(state_dir, "tmp")
    research_dir = os.path.join(state_dir, "research_memos")
    ensure_dir(logs_dir)
    ensure_dir(config_dir)
    ensure_dir(cache_dir)
    ensure_dir(temp_root_dir)
    ensure_dir(research_dir)
    ensure_dir(model_dir(state_dir))

    results_path = os.path.join(state_dir, "results.tsv")
    state_path = os.path.join(state_dir, "state.json")
    best_config_path = os.path.join(state_dir, "best_config.json")
    lock_path = os.path.join(state_dir, LOCK_FILENAME)
    state = None
    run_rows = []

    try:
        with advisory_lock(lock_path):
            ensure_free_space(state_dir, args.min_free_gb, "starting autoresearch")
            ensure_results_file(results_path)

            default_state = {
                "iteration": 0,
                "best": {
                    "name": "seed",
                    "best_val_auc": 0.0,
                    "best_val_loss": None,
                    "config": deepcopy(DEFAULT_CONFIG),
                },
                "research": initialize_research_state(),
            }
            state = load_json(
                state_path,
                default_state,
            )
            result_rows = read_results_rows(results_path)
            state.setdefault("iteration", 0)
            state.setdefault("best", deepcopy(default_state["best"]))
            state["research"] = merge_research_state_defaults(state.get("research"))
            state["best"]["config"] = sanitize_search_config(state["best"].get("config", {}))
            state["best"]["config"] = apply_runtime_overrides(state["best"]["config"], args)
            state["best"]["config"] = persistent_config_snapshot(state["best"]["config"])
            state = backfill_best_metadata(state, result_rows)
            state = recover_best_from_history(state, result_rows)
            current_best_model_path = best_model_path(state_dir)
            keep_paths = [current_best_model_path] if os.path.exists(current_best_model_path) else []
            prune_model_dir(state_dir, keep_paths=keep_paths)
            save_json(best_config_path, state["best"]["config"])
            save_json(state_path, state)

            rng = random.Random(args.seed + int(state["iteration"]))
            run_baseline_first = is_fresh_state(state, result_rows)
            seen_signatures = load_seen_signatures(config_dir)
            seen_signatures.add(config_signature(state["best"]["config"]))

            for _ in range(args.iterations):
                state["iteration"] += 1
                parent = deepcopy(state["best"])
                parent["research"] = state["research"]
                try:
                    if run_baseline_first:
                        candidate_config, mutations, prior = make_baseline_candidate(args)
                        controller_proposal = {
                            "config": candidate_config,
                            "mutations": list(mutations) + ["controller:baseline_bootstrap"],
                            "controller_name": "baseline_bootstrap",
                            "analysis": ["Fresh state detected, so the loop is running a baseline anchor experiment before invoking the research controller."],
                            "hypothesis": "Establish a stable baseline before deeper research-guided iterations.",
                            "sources": [],
                            "context": {},
                        }
                        mutations = list(controller_proposal["mutations"])
                    else:
                        controller_proposal = propose_next_experiment(
                            iteration=state["iteration"],
                            parent=parent,
                            result_rows=result_rows + run_rows,
                            research_state=state["research"],
                            search_space=SEARCH_SPACE,
                            default_config=DEFAULT_CONFIG,
                            research_priors=RESEARCH_PRIORS,
                            seen_signatures=seen_signatures,
                            max_duplicate_retries=MAX_DUPLICATE_RETRIES,
                            seed=args.seed + int(state["iteration"]),
                            agent_command=args.research_agent_cmd,
                        )
                        candidate_config = sanitize_search_config(controller_proposal["config"])
                        candidate_config = apply_runtime_overrides(candidate_config, args)
                        candidate_config = sanitize_search_config(candidate_config)
                        mutations = list(controller_proposal.get("mutations") or [])
                        prior = {"name": controller_proposal.get("controller_name", "research_agent") }
                    run_baseline_first = False
                    init_checkpoint = best_checkpoint_for_parent(state_dir, parent)
                    candidate_name, config_path, log_path, returncode, summary, log_text = run_candidate(
                        script_dir=script_dir,
                        logs_dir=logs_dir,
                        config_dir=config_dir,
                        cache_dir=cache_dir,
                        temp_root_dir=temp_root_dir,
                        iteration=state["iteration"],
                        candidate=candidate_config,
                        data_root=args.data_root,
                        model_storage_root=state_dir,
                        init_checkpoint=init_checkpoint,
                    )
                    if should_retry_without_pretrained(returncode, log_text, candidate_config):
                        discard_candidate_model(state_dir, candidate_name)
                        candidate_config["pretrained"] = 0
                        mutations.append("pretrained:1->0(auto-retry)")
                        candidate_name, config_path, log_path, returncode, summary, log_text = run_candidate(
                            script_dir=script_dir,
                            logs_dir=logs_dir,
                            config_dir=config_dir,
                            cache_dir=cache_dir,
                            temp_root_dir=temp_root_dir,
                            iteration=state["iteration"],
                            candidate=candidate_config,
                            data_root=args.data_root,
                            model_storage_root=state_dir,
                            init_checkpoint=init_checkpoint,
                        )

                    candidate_auc = float_or_default(summary["best_val_auc"])
                    status = candidate_status(
                        parent=parent,
                        candidate_auc=candidate_auc,
                        candidate_config=candidate_config,
                        summary=summary,
                        returncode=returncode,
                    )
                except Exception:
                    prior = {"name": "controller_exception"}
                    controller_proposal = {
                        "config": parent["config"],
                        "mutations": ["controller_exception"],
                        "controller_name": "controller_exception",
                        "analysis": ["The research controller failed before launching the candidate. The loop recorded the traceback and kept the current best configuration."],
                        "hypothesis": "Controller failure fallback.",
                        "sources": [],
                        "context": {},
                    }
                    mutations = ["controller_exception"]
                    candidate_config = sanitize_search_config(parent["config"])
                    candidate_name = f"iter{state['iteration']:03d}_controller_exception"
                    config_path = os.path.join(config_dir, f"{candidate_name}.json")
                    log_path = os.path.join(logs_dir, f"{candidate_name}.log")
                    save_json(config_path, candidate_config)
                    with open(log_path, "w") as handle:
                        handle.write(traceback.format_exc())
                    returncode = 1
                    summary = {}
                    candidate_auc = 0.0
                    status = "crash"

                if status == "keep":
                    promote_candidate_model(
                        state_dir,
                        candidate_name,
                        candidate_config,
                        candidate_auc,
                    )
                else:
                    discard_candidate_model(state_dir, candidate_name)

                research_memo = controller_memo_path(research_dir, candidate_name)
                write_research_memo(
                    research_memo,
                    controller_proposal,
                    outcome=summarize_outcome(
                        status=status,
                        candidate_name=candidate_name,
                        candidate_auc=candidate_auc,
                        summary=summary,
                        config_path=config_path,
                        log_path=log_path,
                    ),
                )

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iteration": state["iteration"],
                    "candidate_name": candidate_name,
                    "parent_name": parent["name"],
                    "status": status,
                    "best_val_auc": f"{candidate_auc:.6f}",
                    "best_val_loss": summary.get("best_val_loss") or "nan",
                    "training_seconds": summary.get("training_seconds") or "0",
                    "epochs_ran": summary.get("epochs_ran") or "0",
                    "num_params_M": summary.get("num_params_M") or "0",
                    "device": summary.get("device") or "unknown",
                    "model_complexity": summary.get("model_complexity") or "",
                    "mutations": ", ".join(mutations),
                    "config_path": config_path,
                    "log_file": log_path,
                    "research_memo": research_memo,
                }
                append_result(results_path, row)
                run_rows.append(row)
                seen_signatures.add(config_signature(candidate_config))
                if prior["name"] in state["research"]:
                    state["research"][prior["name"]]["trials"] += 1

                if status == "keep":
                    persistent_candidate_config = persistent_config_snapshot(candidate_config)
                    state["best"] = {
                        "name": candidate_name,
                        "best_val_auc": candidate_auc,
                        "best_val_loss": float_or_default(summary["best_val_loss"], None),
                        "num_params_M": float_or_default(summary["num_params_M"], None),
                        "complexity_score": config_complexity_score(persistent_candidate_config),
                        "config": persistent_candidate_config,
                    }
                    save_json(best_config_path, persistent_candidate_config)
                    if prior["name"] in state["research"]:
                        state["research"][prior["name"]]["keeps"] += 1

                save_json(state_path, state)
                print(
                    f"iteration={state['iteration']} candidate={candidate_name} "
                    f"status={status} val_auc={candidate_auc:.6f}"
                )
                ensure_free_space(state_dir, args.min_free_gb, "continuing autoresearch")

            write_latest_run_manifest(state_dir, run_rows)
            write_best_only_manifest(state_dir, state["best"]["name"])
            if args.summary_file:
                write_markdown_summary(args.summary_file, state, run_rows)
    except LowDiskSpaceError as exc:
        message = str(exc)
        print(message)
        if args.summary_file and state is not None:
            write_markdown_summary(args.summary_file, state, run_rows, notes=[message])
        else:
            write_lock_summary(args.summary_file, message)
        return
    except LockUnavailableError as exc:
        message = f"Skipped run because {exc}."
        print(message)
        write_lock_summary(args.summary_file, message)
        return


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
    parser.add_argument("--min_free_gb", type=float, default=40.0)
    parser.add_argument(
        "--research_agent_cmd",
        type=str,
        default=os.environ.get("AUTORESEARCH_RESEARCH_AGENT_CMD"),
        help="Optional external command that receives JSON context on stdin and returns a JSON proposal for the next experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
