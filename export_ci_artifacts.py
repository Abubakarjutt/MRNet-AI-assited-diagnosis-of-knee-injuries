#!/usr/bin/env python3
"""Export autoresearch artifacts into a CI-friendly output directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def copy_if_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def export_latest_logs(state_dir: Path, output_dir: Path) -> int:
    manifest_path = state_dir / "latest_run_files.json"
    if not manifest_path.is_file():
        return 0

    copied = 0
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for row in manifest.get("rows", []):
        log_file = row.get("log_file")
        if not log_file:
            continue
        src = Path(log_file).expanduser()
        if not src.is_file():
            continue
        shutil.copy2(src, logs_dir / src.name)
        copied += 1

    print(f"Exported {copied} latest-run log files to {logs_dir}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, help="Autoresearch state directory")
    parser.add_argument("--output-dir", required=True, help="CI artifact output directory")
    args = parser.parse_args()

    state_dir = Path(args.state_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    copied += copy_if_file(state_dir / "results.tsv", output_dir / "results.tsv")
    copied += copy_if_file(state_dir / "best_config.json", output_dir / "best_config.json")
    copied += copy_if_file(state_dir / "latest_run_files.json", output_dir / "latest_run_files.json")
    copied += export_latest_logs(state_dir, output_dir)

    print(f"Exported {copied} autoresearch artifacts into {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CI guardrail
        print(f"artifact export failed: {exc}", file=sys.stderr)
        raise
