#!/usr/bin/env python3
"""Reorganize existing LIBERO eval videos into a GR00T-style failure-only tree.

Old OpenPI layout:
  data/<eval_name>/videos/seed7_steps10/rollout_<task>_failure_3.mp4

New layout:
  data/rollouts/<model>/<suite>/h10/seed7--episode=3--success=False--task=<task>.mp4

By default this script hard-links files, so it is fast and avoids duplicating
video bytes. Use --mode copy if hard links are not desired.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


VIDEO_RE = re.compile(r"^rollout_(?P<task>.+)_(?P<status>success|failure)_(?P<episode>\d+)\.mp4$")
RUN_RE = re.compile(r"^seed(?P<seed>\d+)_(?:steps|h)(?P<horizon>\d+)$")


@dataclass(frozen=True)
class FailureVideo:
    source: Path
    dest: Path
    model: str
    suite: str
    seed: int
    horizon: int
    episode: int
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/rollouts-style failure-only links/copies from existing data/*/videos folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--mode", choices=("link", "copy"), default="link")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Defaults to <output-root>/failure_video_manifest.csv")
    return parser.parse_args()


def processed_task(task: str) -> str:
    return task.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]


def model_and_suite(eval_name: str) -> tuple[str, str] | None:
    if eval_name.endswith("_libero10"):
        return eval_name.removesuffix("_libero10"), "libero_10"
    if eval_name.endswith("_libero_pro"):
        # Existing OpenPI libero_pro folders correspond to object perturbation
        # runs in their status JSONs.
        return eval_name.removesuffix("_libero_pro"), "libero_10_pertobject"
    return None


def iter_failure_videos(data_root: Path, output_root: Path) -> list[FailureVideo]:
    failures: list[FailureVideo] = []
    for videos_dir in sorted(data_root.glob("*/videos")):
        parsed = model_and_suite(videos_dir.parent.name)
        if parsed is None:
            continue
        model, suite = parsed
        for run_dir in sorted(path for path in videos_dir.iterdir() if path.is_dir()):
            run_match = RUN_RE.match(run_dir.name)
            if not run_match:
                continue
            seed = int(run_match.group("seed"))
            horizon = int(run_match.group("horizon"))
            for source in sorted(run_dir.glob("*.mp4")):
                video_match = VIDEO_RE.match(source.name)
                if not video_match or video_match.group("status") != "failure":
                    continue
                task = video_match.group("task")
                episode = int(video_match.group("episode"))
                dest_dir = output_root / model / suite / f"h{horizon}"
                dest = dest_dir / (
                    f"seed{seed}--episode={episode}--success=False"
                    f"--task={processed_task(task)}.mp4"
                )
                failures.append(
                    FailureVideo(
                        source=source,
                        dest=dest,
                        model=model,
                        suite=suite,
                        seed=seed,
                        horizon=horizon,
                        episode=episode,
                        task=task,
                    )
                )
    return failures


def materialize(video: FailureVideo, mode: str, overwrite: bool, dry_run: bool) -> str:
    if video.dest.exists():
        if not overwrite:
            return "exists"
        if not dry_run:
            video.dest.unlink()
    if dry_run:
        return "dry_run"
    video.dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(video.source, video.dest)
        return "copied"
    os.link(video.source, video.dest)
    return "linked"


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest or args.output_root / "failure_video_manifest.csv"
    failures = iter_failure_videos(args.data_root, args.output_root)
    manifest_rows: list[dict[str, str | int]] = []
    counts: dict[str, int] = {}
    for video in failures:
        status = materialize(video, args.mode, args.overwrite, args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        manifest_rows.append(
            {
                "status": status,
                "model": video.model,
                "suite": video.suite,
                "seed": video.seed,
                "horizon": video.horizon,
                "episode": video.episode,
                "task": video.task,
                "source": video.source,
                "dest": video.dest,
            }
        )
    write_manifest(manifest_path, manifest_rows)
    print(f"Found {len(failures)} failure videos")
    print("Actions:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"Wrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
