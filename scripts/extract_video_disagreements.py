#!/usr/bin/env python3
"""List eval videos where two LIBERO model runs disagree on success/failure.

The evaluator writes videos like:

  data/<eval_name>/videos/seed100_steps10/rollout_<task>_success_3.mp4

This script compares matching videos by suite, seed, horizon, task slug, and
episode index, then emits rows for cases where one model succeeded and the
other failed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


VIDEO_RE = re.compile(r"^rollout_(?P<task>.+)_(?P<status>success|failure)_(?P<episode>\d+)\.mp4$")
RUN_DIR_RE = re.compile(r"^seed(?P<seed>\d+)_(?:steps|h)(?P<horizon>\d+)$")
SUITES = ("libero10", "libero_pro")


@dataclass(frozen=True)
class CaseKey:
    suite: str
    seed: int
    horizon: int
    task: str
    episode: int


@dataclass(frozen=True)
class VideoCase:
    success: bool
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video paths where two LIBERO eval runs disagree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-a", required=True, help="Model/eval name, e.g. pi05_lora_gated_film_skill_router_joint")
    parser.add_argument("--model-b", required=True, help="Model/eval name, e.g. pi05_libero_lora_skill_router_joint")
    parser.add_argument(
        "--suite",
        choices=("libero10", "libero_pro", "both"),
        default="both",
        help="Eval suite to compare.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root containing eval video folders.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path. Defaults to stdout only.")
    parser.add_argument(
        "--only",
        choices=("both", "a_success", "b_success"),
        default="both",
        help="Filter disagreements by which model succeeded.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to print/write; 0 means no limit.")
    return parser.parse_args()


def candidate_eval_names(model: str, suite: str) -> list[str]:
    if model.endswith(f"_{suite}"):
        return [model]
    return [f"{model}_{suite}", model]


def resolve_video_dir(data_root: Path, model: str, suite: str) -> Path:
    candidates = [data_root / name / "videos" for name in candidate_eval_names(model, suite)]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    joined = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No video directory found for model={model!r}, suite={suite!r}. Tried:\n  {joined}")


def load_cases(video_dir: Path, suite: str) -> dict[CaseKey, VideoCase]:
    cases: dict[CaseKey, VideoCase] = {}
    for run_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
        run_match = RUN_DIR_RE.match(run_dir.name)
        if not run_match:
            continue
        seed = int(run_match.group("seed"))
        horizon = int(run_match.group("horizon"))

        for video_path in sorted(run_dir.glob("*.mp4")):
            video_match = VIDEO_RE.match(video_path.name)
            if not video_match:
                continue
            key = CaseKey(
                suite=suite,
                seed=seed,
                horizon=horizon,
                task=video_match.group("task"),
                episode=int(video_match.group("episode")),
            )
            cases[key] = VideoCase(
                success=video_match.group("status") == "success",
                path=video_path.resolve(),
            )
    return cases


def row_for(key: CaseKey, a: VideoCase, b: VideoCase, model_a: str, model_b: str) -> dict[str, str | int]:
    winner = model_a if a.success else model_b
    return {
        "suite": key.suite,
        "seed": key.seed,
        "horizon": key.horizon,
        "task": key.task,
        "episode": key.episode,
        f"{model_a}_status": "success" if a.success else "failure",
        f"{model_b}_status": "success" if b.success else "failure",
        "winner": winner,
        f"{model_a}_video": str(a.path),
        f"{model_b}_video": str(b.path),
    }


def main() -> int:
    args = parse_args()
    suites = SUITES if args.suite == "both" else (args.suite,)
    rows: list[dict[str, str | int]] = []

    for suite in suites:
        a_dir = resolve_video_dir(args.data_root, args.model_a, suite)
        b_dir = resolve_video_dir(args.data_root, args.model_b, suite)
        a_cases = load_cases(a_dir, suite)
        b_cases = load_cases(b_dir, suite)
        shared_keys = sorted(set(a_cases) & set(b_cases), key=lambda k: (k.suite, k.seed, k.horizon, k.task, k.episode))

        for key in shared_keys:
            a_case = a_cases[key]
            b_case = b_cases[key]
            if a_case.success == b_case.success:
                continue
            if args.only == "a_success" and not a_case.success:
                continue
            if args.only == "b_success" and not b_case.success:
                continue
            rows.append(row_for(key, a_case, b_case, args.model_a, args.model_b))
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

        missing_a = len(set(b_cases) - set(a_cases))
        missing_b = len(set(a_cases) - set(b_cases))
        print(
            f"[{suite}] shared={len(shared_keys)} disagreements_so_far={len(rows)} "
            f"missing_in_a={missing_a} missing_in_b={missing_b}",
            file=sys.stderr,
        )

    if not rows:
        print("No success/failure disagreements found.", file=sys.stderr)
        return 0

    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
