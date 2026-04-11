#!/usr/bin/env python3
"""Prune stale self-hosted GitHub Actions runner diagnostics."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def infer_runner_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp:
        temp_path = Path(runner_temp).expanduser().resolve()
        if temp_path.name == "_temp" and temp_path.parent.name == "_work":
            return temp_path.parent.parent

    return (Path.home() / "actions-runner-mrnet-autoresearch").resolve()


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def prune_keep_newest(files: list[Path], keep: int) -> tuple[int, int]:
    removed_count = 0
    removed_bytes = 0

    for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[keep:]:
        removed_bytes += file_size(path)
        try:
            path.unlink()
            removed_count += 1
        except FileNotFoundError:
            continue

    return removed_count, removed_bytes


def prune_directory_entries(directory: Path, keep: int) -> tuple[int, int]:
    if not directory.is_dir():
        return 0, 0
    files = [path for path in directory.iterdir() if path.is_file()]
    return prune_keep_newest(files, keep)


def prune_named_logs(diag_dir: Path, prefix: str, keep: int) -> tuple[int, int]:
    files = sorted(diag_dir.glob(f"{prefix}_*.log"))
    return prune_keep_newest(files, keep)


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-root", default=None, help="Path to the self-hosted runner root")
    parser.add_argument("--keep-runner-logs", type=int, default=3)
    parser.add_argument("--keep-worker-logs", type=int, default=6)
    parser.add_argument("--keep-pages", type=int, default=64)
    parser.add_argument("--keep-blocks", type=int, default=64)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    runner_root = infer_runner_root(args.runner_root)
    diag_dir = runner_root / "_diag"
    pages_dir = diag_dir / "pages"
    blocks_dir = diag_dir / "blocks"

    total_removed_count = 0
    total_removed_bytes = 0

    for removed_count, removed_bytes in [
        prune_named_logs(diag_dir, "Runner", args.keep_runner_logs),
        prune_named_logs(diag_dir, "Worker", args.keep_worker_logs),
        prune_directory_entries(pages_dir, args.keep_pages),
        prune_directory_entries(blocks_dir, args.keep_blocks),
    ]:
        total_removed_count += removed_count
        total_removed_bytes += removed_bytes

    free_bytes = shutil.disk_usage(runner_root).free
    print(f"runner_root={runner_root}")
    print(f"removed_files={total_removed_count}")
    print(f"freed_bytes={total_removed_bytes}")
    print(f"freed_human={format_gb(total_removed_bytes)}")
    print(f"free_bytes={free_bytes}")
    print(f"free_human={format_gb(free_bytes)}")

    min_free_bytes = int(args.min_free_gb * (1024 ** 3))
    if free_bytes < min_free_bytes:
        raise SystemExit(
            f"Runner free space {format_gb(free_bytes)} is below the required {args.min_free_gb:.1f} GiB."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
