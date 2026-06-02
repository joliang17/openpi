#!/usr/bin/env python3
"""Create paged grid videos for success/failure disagreement review.

Input rows should come from scripts/extract_video_disagreements.py. Each output
page stacks multiple disagreement cases vertically, with model A on the left and
model B on the right for easy side-by-side comparison.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CSV = Path("results_csv/pi05_lora_gated_film_vs_lora_skill_router_video_disagreements.csv")
DEFAULT_OUT_DIR = Path("data/video_grids/lora_gated_vs_lora_router")
GATED_MODEL = "pi05_lora_gated_film_skill_router_joint"
BASELINE_MODEL = "pi05_libero_lora_skill_router_joint"


@dataclass(frozen=True)
class DisagreementRow:
    suite: str
    seed: int
    horizon: int
    task: str
    episode: int
    model_a_status: str
    model_b_status: str
    winner: str
    model_a_video: Path
    model_b_video: Path
    raw_index: int


@dataclass(frozen=True)
class FailureClip:
    row: DisagreementRow
    direction: str
    failed_model: str
    video: Path


@dataclass
class VideoReader:
    path: Path
    cap: cv2.VideoCapture
    frame_count: int
    fps: float
    frame_stride: int
    last_frame: np.ndarray | None = None
    exhausted: bool = False

    @classmethod
    def open(cls, path: Path, output_fps: float, speedup: float = 1.0) -> "VideoReader":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        frame_stride = max(1, int(round((fps / output_fps) * speedup)))
        sampled_frame_count = max(1, math.ceil(frame_count / frame_stride))
        return cls(path=path, cap=cap, frame_count=sampled_frame_count, fps=fps, frame_stride=frame_stride)

    def read_hold_last(self) -> np.ndarray:
        if not self.exhausted:
            ok, frame = self.cap.read()
            if ok:
                self.last_frame = frame
                for _ in range(self.frame_stride - 1):
                    skipped, _ = self.cap.read()
                    if not skipped:
                        self.exhausted = True
                        break
                return frame
            self.exhausted = True
        if self.last_frame is None:
            self.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        return self.last_frame

    def release(self) -> None:
        self.cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paged grid MP4s comparing disagreement videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Disagreement CSV with video path columns.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for grid videos.")
    parser.add_argument("--suite", choices=("both", "libero10", "libero_pro"), default="both")
    parser.add_argument(
        "--winner",
        choices=("both", "gated", "baseline"),
        default="both",
        help="Filter by which model succeeded.",
    )
    parser.add_argument("--rows-per-page", type=int, default=4, help="Disagreement cases per output video.")
    parser.add_argument("--width", type=int, default=480, help="Width of each source video cell.")
    parser.add_argument(
        "--cell-width",
        type=int,
        help="Alias for --width. Useful for task-grouped single-clip grid cells.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=0,
        help="Columns for --group-by-task. 0 chooses a wide layout automatically.",
    )
    parser.add_argument(
        "--group-by-task",
        action="store_true",
        help="Create one failure-only grid video per task and direction: gate_wins / gate_loses.",
    )
    parser.add_argument(
        "--max-clips-per-video",
        type=int,
        default=0,
        help="Optional split size for task-grouped outputs; 0 keeps each task/direction in one video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output FPS. Source frames are sampled down to roughly this rate.",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Additional frame skipping while keeping output FPS. 2.0 makes videos about 2x faster.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("/opt/local/stow/ffmpeg-7.1/bin/ffmpeg"),
        help="ffmpeg binary for H.264 conversion in task-grouped mode.",
    )
    parser.add_argument("--crf", type=int, default=23, help="libx264 CRF for ffmpeg conversion.")
    parser.add_argument("--preset", default="veryfast", help="libx264 preset for ffmpeg conversion.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary OpenCV MP4s before H.264 conversion.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum disagreement rows to render; 0 means all.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest CSV path. Defaults inside output dir.")
    parser.add_argument("--report-md", type=Path, help="Optional Markdown report path. Defaults inside output dir.")
    parser.add_argument("--report-html", type=Path, help="Optional HTML report path. Defaults inside output dir.")
    return parser.parse_args()


def find_column(fieldnames: Iterable[str], suffix: str, contains: str | None = None) -> str:
    matches = [name for name in fieldnames if name.endswith(suffix) and (contains is None or contains in name)]
    if not matches:
        raise ValueError(f"Could not find column ending with {suffix!r} containing {contains!r}")
    return matches[0]


def load_rows(csv_path: Path, suite_filter: str, winner_filter: str, limit: int) -> tuple[list[DisagreementRow], str, str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        model_a_video_col = find_column(reader.fieldnames, "_video", GATED_MODEL)
        model_b_video_col = find_column(reader.fieldnames, "_video", BASELINE_MODEL)
        model_a_status_col = find_column(reader.fieldnames, "_status", GATED_MODEL)
        model_b_status_col = find_column(reader.fieldnames, "_status", BASELINE_MODEL)

        rows: list[DisagreementRow] = []
        for raw_index, row in enumerate(reader, start=2):
            if suite_filter != "both" and row["suite"] != suite_filter:
                continue
            winner = row["winner"]
            if winner_filter == "gated" and winner != GATED_MODEL:
                continue
            if winner_filter == "baseline" and winner != BASELINE_MODEL:
                continue

            item = DisagreementRow(
                suite=row["suite"],
                seed=int(row["seed"]),
                horizon=int(row["horizon"]),
                task=row["task"],
                episode=int(row["episode"]),
                model_a_status=row[model_a_status_col],
                model_b_status=row[model_b_status_col],
                winner=winner,
                model_a_video=Path(row[model_a_video_col]),
                model_b_video=Path(row[model_b_video_col]),
                raw_index=raw_index,
            )
            rows.append(item)
            if limit and len(rows) >= limit:
                break

    rows.sort(key=lambda r: (r.suite, r.seed, r.horizon, r.task, r.episode))
    return rows, model_a_video_col.removesuffix("_video"), model_b_video_col.removesuffix("_video")


def short_model_name(model: str) -> str:
    if model == GATED_MODEL:
        return "lora_gated_film"
    if model == BASELINE_MODEL:
        return "lora_skill_router"
    return model.replace("pi05_", "").replace("_joint", "")


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = re.split(r"([_ ])", text)
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current.rstrip("_ "))
            current = word.lstrip("_ ")
    if current:
        lines.append(current.rstrip("_ "))
    return lines[:2]


def slugify(text: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len].rstrip("_") or "task"


def task_label(text: str) -> str:
    return text.replace("_", " ")


def suite_label(suite: str) -> str:
    if suite == "libero10":
        return "libero10"
    if suite == "libero_pro":
        return "liberopro_object"
    return slugify(suite)


def direction_label(direction: str) -> str:
    if direction == "gate_wins":
        return "gated_win"
    if direction == "gate_loses":
        return "gated_lose"
    return direction


def make_label(
    width: int,
    height: int,
    row: DisagreementRow,
    model_a_name: str,
    model_b_name: str,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small = ImageFont.load_default()

    left_color = (32, 140, 70) if row.model_a_status == "success" else (170, 45, 45)
    right_color = (32, 140, 70) if row.model_b_status == "success" else (170, 45, 45)
    half = width // 2
    draw.rectangle((0, 0, half - 1, height - 1), outline=left_color, width=4)
    draw.rectangle((half, 0, width - 1, height - 1), outline=right_color, width=4)

    title = f"{row.suite} | seed {row.seed} | h{row.horizon} | ep {row.episode}"
    task_lines = wrap_text(draw, row.task, small, width - 24)
    draw.text((12, 8), title, fill=(235, 235, 235), font=font)
    for i, line in enumerate(task_lines):
        draw.text((12, 26 + i * 13), line, fill=(220, 220, 220), font=small)

    draw.text((12, height - 18), f"{short_model_name(model_a_name)}: {row.model_a_status}", fill=left_color, font=small)
    draw.text(
        (half + 12, height - 18),
        f"{short_model_name(model_b_name)}: {row.model_b_status}",
        fill=right_color,
        font=small,
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def make_clip_label(width: int, height: int, clip: FailureClip) -> np.ndarray:
    image = Image.new("RGB", (width, height), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    color = (170, 45, 45)
    draw.rectangle((0, 0, width - 1, height - 1), outline=color, width=4)

    row = clip.row
    title = f"{clip.direction} | failed: {short_model_name(clip.failed_model)}"
    meta = f"{row.suite} | seed {row.seed} | h{row.horizon} | ep {row.episode}"
    draw.text((10, 7), title, fill=color, font=font)
    draw.text((10, 23), meta, fill=(235, 235, 235), font=small)
    for i, line in enumerate(wrap_text(draw, task_label(row.task), small, width - 20)):
        draw.text((10, 39 + i * 13), line, fill=(220, 220, 220), font=small)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    height, old_width = frame.shape[:2]
    new_height = max(1, int(round(height * width / old_width)))
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    current = frame.shape[0]
    if current == height:
        return frame
    if current > height:
        return frame[:height, :, :]
    top = (height - current) // 2
    bottom = height - current - top
    return cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def compose_case_frame(
    left: np.ndarray,
    right: np.ndarray,
    label: np.ndarray,
    cell_width: int,
    gap: int,
    row_video_height: int,
) -> np.ndarray:
    left = pad_to_height(resize_to_width(left, cell_width), row_video_height)
    right = pad_to_height(resize_to_width(right, cell_width), row_video_height)
    gutter = np.full((row_video_height, gap, 3), 18, dtype=np.uint8)
    videos = np.concatenate([left, gutter, right], axis=1)
    return np.concatenate([label, videos], axis=0)


def choose_auto_cols(num_clips: int) -> int:
    cols = max(2, math.ceil(math.sqrt(num_clips)))
    while cols < math.ceil(num_clips / cols):
        cols += 1
    return cols


def clip_for_row(row: DisagreementRow) -> FailureClip:
    if row.winner == GATED_MODEL:
        return FailureClip(
            row=row,
            direction="gate_wins",
            failed_model=BASELINE_MODEL,
            video=row.model_b_video,
        )
    if row.winner == BASELINE_MODEL:
        return FailureClip(
            row=row,
            direction="gate_loses",
            failed_model=GATED_MODEL,
            video=row.model_a_video,
        )
    raise ValueError(f"Unexpected winner: {row.winner}")


def compose_clip_frame(frame: np.ndarray, label: np.ndarray, cell_width: int, cell_video_height: int) -> np.ndarray:
    frame = pad_to_height(resize_to_width(frame, cell_width), cell_video_height)
    return np.concatenate([label, frame], axis=0)


def render_failure_grid(
    clips: list[FailureClip],
    output_path: Path,
    cell_width: int,
    cols: int,
    fps: float,
    speedup: float,
) -> None:
    if not clips:
        raise ValueError("render_failure_grid requires at least one clip")
    gap = 12
    label_height = 68
    readers: list[VideoReader] = []
    try:
        max_frames = 1
        max_video_height = 1
        for clip in clips:
            if not clip.video.exists():
                raise FileNotFoundError(clip.video)
            reader = VideoReader.open(clip.video, fps, speedup)
            readers.append(reader)
            max_frames = max(max_frames, reader.frame_count)
            width = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cell_width
            height = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cell_width
            max_video_height = max(max_video_height, int(round(height * cell_width / width)))

        rows = math.ceil(len(clips) / cols)
        cell_height = label_height + max_video_height
        frame_width = cols * cell_width + (cols - 1) * gap
        frame_height = rows * cell_height + (rows - 1) * gap
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for: {output_path}")

        labels = [make_clip_label(cell_width, label_height, clip) for clip in clips]
        blank_cell = np.full((cell_height, cell_width, 3), 18, dtype=np.uint8)
        v_gap = np.full((cell_height, gap, 3), 18, dtype=np.uint8)
        h_gap = np.full((gap, frame_width, 3), 18, dtype=np.uint8)
        for _ in range(max_frames):
            cells = [
                compose_clip_frame(readers[idx].read_hold_last(), labels[idx], cell_width, max_video_height)
                for idx in range(len(readers))
            ]
            while len(cells) < rows * cols:
                cells.append(blank_cell)

            row_frames: list[np.ndarray] = []
            for row_idx in range(rows):
                parts: list[np.ndarray] = []
                for col_idx in range(cols):
                    if col_idx:
                        parts.append(v_gap)
                    parts.append(cells[row_idx * cols + col_idx])
                row_frames.append(np.concatenate(parts, axis=1))

            page_parts: list[np.ndarray] = []
            for idx, row_frame in enumerate(row_frames):
                if idx:
                    page_parts.append(h_gap)
                page_parts.append(row_frame)
            writer.write(np.concatenate(page_parts, axis=0))
        writer.release()
    finally:
        for reader in readers:
            reader.release()


def resolve_ffmpeg(ffmpeg: Path) -> Path:
    if ffmpeg.exists() and ffmpeg.is_file():
        return ffmpeg
    found = shutil.which(str(ffmpeg))
    if found:
        return Path(found)
    raise FileNotFoundError(f"ffmpeg not found at {ffmpeg}. Try: module load ffmpeg/7.1")


def convert_to_h264(temp_path: Path, final_path: Path, ffmpeg: Path, crf: int, preset: str) -> None:
    ffmpeg_path = resolve_ffmpeg(ffmpeg)
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temp_path),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_path),
    ]
    subprocess.run(cmd, check=True)


def render_page(
    rows: list[DisagreementRow],
    output_path: Path,
    model_a_name: str,
    model_b_name: str,
    cell_width: int,
    fps: float,
) -> None:
    gap = 12
    label_height = 64
    row_gap = 16
    readers: list[tuple[VideoReader, VideoReader]] = []
    try:
        max_frames = 1
        max_video_height = 1
        for row in rows:
            if not row.model_a_video.exists():
                raise FileNotFoundError(row.model_a_video)
            if not row.model_b_video.exists():
                raise FileNotFoundError(row.model_b_video)
            left = VideoReader.open(row.model_a_video, fps)
            right = VideoReader.open(row.model_b_video, fps)
            readers.append((left, right))
            max_frames = max(max_frames, left.frame_count, right.frame_count)

            for reader in (left, right):
                width = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cell_width
                height = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cell_width
                max_video_height = max(max_video_height, int(round(height * cell_width / width)))

        frame_width = cell_width * 2 + gap
        case_height = label_height + max_video_height
        frame_height = case_height * len(rows) + row_gap * (len(rows) - 1)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for: {output_path}")

        labels = [make_label(frame_width, label_height, row, model_a_name, model_b_name) for row in rows]
        separator = np.full((row_gap, frame_width, 3), 18, dtype=np.uint8)
        for _ in range(max_frames):
            case_frames: list[np.ndarray] = []
            for idx, (left, right) in enumerate(readers):
                case_frames.append(
                    compose_case_frame(
                        left.read_hold_last(),
                        right.read_hold_last(),
                        labels[idx],
                        cell_width,
                        gap,
                        max_video_height,
                    )
                )
            page_parts: list[np.ndarray] = []
            for idx, case_frame in enumerate(case_frames):
                if idx:
                    page_parts.append(separator)
                page_parts.append(case_frame)
            writer.write(np.concatenate(page_parts, axis=0))
        writer.release()
    finally:
        for left, right in readers:
            left.release()
            right.release()


def page_name(page_index: int, rows: list[DisagreementRow]) -> str:
    suites = sorted({row.suite for row in rows})
    suite_part = suites[0] if len(suites) == 1 else "mixed"
    return f"page_{page_index:03d}_{suite_part}.mp4"


def write_manifest(manifest_path: Path, records: list[dict[str, str | int]]) -> None:
    if not records:
        return
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_reports(md_path: Path, html_path: Path, records: list[dict[str, str | int]]) -> None:
    summary: dict[tuple[str, str, int], dict[str, int]] = {}
    totals: dict[str, dict[str, int]] = {}
    for record in records:
        suite = str(record["suite_dir"])
        task = str(record["task"])
        horizon = int(record["horizon"])
        direction = str(record["direction"])
        key = (suite, task, horizon)
        summary.setdefault(key, {"gated_win": 0, "gated_lose": 0})[direction] += 1
        totals.setdefault(suite, {"gated_win": 0, "gated_lose": 0})[direction] += 1

    md_lines = [
        "# Lora Gated-FiLM vs LoRA Skill Router Failure Review",
        "",
        "Counts are disagreement cases only. `gated_win` means gated-FiLM succeeded while the baseline failed; "
        "`gated_lose` means gated-FiLM failed while the baseline succeeded.",
        "",
        "## Totals",
        "",
        "| Suite | Gated better | Gated worse | Net |",
        "|---|---:|---:|---:|",
    ]
    for suite in sorted(totals):
        wins = totals[suite]["gated_win"]
        losses = totals[suite]["gated_lose"]
        md_lines.append(f"| {suite} | {wins} | {losses} | {wins - losses:+d} |")

    md_lines.extend(
        [
            "",
            "## By Task And Horizon",
            "",
            "| Suite | Horizon | Task | Gated better | Gated worse | Net |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for suite, task, horizon in sorted(summary):
        wins = summary[(suite, task, horizon)]["gated_win"]
        losses = summary[(suite, task, horizon)]["gated_lose"]
        md_lines.append(f"| {suite} | h{horizon} | {task_label(task)} | {wins} | {losses} | {wins - losses:+d} |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n")

    html_rows = []
    for suite, task, horizon in sorted(summary):
        wins = summary[(suite, task, horizon)]["gated_win"]
        losses = summary[(suite, task, horizon)]["gated_lose"]
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(suite)}</td>"
            f"<td>h{horizon}</td>"
            f"<td>{html.escape(task_label(task))}</td>"
            f"<td>{wins}</td>"
            f"<td>{losses}</td>"
            f"<td>{wins - losses:+d}</td>"
            "</tr>"
        )
    html_total_rows = []
    for suite in sorted(totals):
        wins = totals[suite]["gated_win"]
        losses = totals[suite]["gated_lose"]
        html_total_rows.append(
            "<tr>"
            f"<td>{html.escape(suite)}</td>"
            f"<td>{wins}</td>"
            f"<td>{losses}</td>"
            f"<td>{wins - losses:+d}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Lora Gated-FiLM vs LoRA Skill Router Failure Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    td:nth-child(2), td:nth-child(4), td:nth-child(5), td:nth-child(6) {{ text-align: right; }}
  </style>
</head>
<body>
  <h1>Lora Gated-FiLM vs LoRA Skill Router Failure Review</h1>
  <p>Counts are disagreement cases only. <code>gated_win</code> means gated-FiLM succeeded while the baseline failed; <code>gated_lose</code> means gated-FiLM failed while the baseline succeeded.</p>
  <h2>Totals</h2>
  <table>
    <thead><tr><th>Suite</th><th>Gated better</th><th>Gated worse</th><th>Net</th></tr></thead>
    <tbody>{''.join(html_total_rows)}</tbody>
  </table>
  <h2>By Task And Horizon</h2>
  <table>
    <thead><tr><th>Suite</th><th>Horizon</th><th>Task</th><th>Gated better</th><th>Gated worse</th><th>Net</th></tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc)


def chunks(items: list[FailureClip], size: int) -> Iterable[list[FailureClip]]:
    if size <= 0:
        yield items
        return
    for start in range(0, len(items), size):
        yield items[start : start + size]


def render_task_grouped_outputs(
    rows: list[DisagreementRow],
    output_dir: Path,
    manifest_path: Path,
    report_md_path: Path,
    report_html_path: Path,
    cell_width: int,
    cols_arg: int,
    fps: float,
    speedup: float,
    max_clips_per_video: int,
    ffmpeg: Path,
    crf: int,
    preset: str,
    keep_temp: bool,
) -> None:
    grouped: dict[tuple[str, str, str], list[FailureClip]] = {}
    for row in rows:
        clip = clip_for_row(row)
        grouped.setdefault((row.suite, row.task, clip.direction), []).append(clip)

    manifest_records: list[dict[str, str | int]] = []
    for (suite, task, direction), clips_for_group in sorted(grouped.items(), key=lambda item: item[0]):
        clips_for_group.sort(key=lambda clip: (clip.row.suite, clip.row.seed, clip.row.horizon, clip.row.episode))
        task_dir = output_dir / suite_label(suite) / slugify(task)
        task_dir.mkdir(parents=True, exist_ok=True)

        for part_index, part_clips in enumerate(chunks(clips_for_group, max_clips_per_video)):
            suffix = f"_part{part_index:02d}" if max_clips_per_video > 0 else ""
            final_path = task_dir / f"{direction_label(direction)}{suffix}.mp4"
            temp_path = task_dir / f".{direction_label(direction)}{suffix}.opencv_tmp.mp4"
            cols = cols_arg if cols_arg > 0 else choose_auto_cols(len(part_clips))
            render_failure_grid(part_clips, temp_path, cell_width, cols, fps, speedup)
            convert_to_h264(temp_path, final_path, ffmpeg, crf, preset)
            if not keep_temp:
                temp_path.unlink(missing_ok=True)
            print(f"Wrote {final_path} ({len(part_clips)} failure clips, cols={cols})")

            for grid_index, clip in enumerate(part_clips):
                row = clip.row
                manifest_records.append(
                    {
                        "task_video": final_path.resolve(),
                        "suite": row.suite,
                        "suite_dir": suite_label(row.suite),
                        "task": row.task,
                        "direction": direction_label(clip.direction),
                        "grid_index": grid_index,
                        "csv_row": row.raw_index,
                        "seed": row.seed,
                        "horizon": row.horizon,
                        "episode": row.episode,
                        "failed_model": clip.failed_model,
                        "failure_video": clip.video,
                        "winner": row.winner,
                        f"{GATED_MODEL}_status": row.model_a_status,
                        f"{BASELINE_MODEL}_status": row.model_b_status,
                        f"{GATED_MODEL}_video": row.model_a_video,
                        f"{BASELINE_MODEL}_video": row.model_b_video,
                    }
                )

    write_manifest(manifest_path, manifest_records)
    write_reports(report_md_path, report_html_path, manifest_records)
    print(f"Wrote manifest {manifest_path} ({len(manifest_records)} failure clips)")
    print(f"Wrote report {report_md_path}")
    print(f"Wrote report {report_html_path}")


def main() -> int:
    args = parse_args()
    if args.rows_per_page <= 0:
        raise ValueError("--rows-per-page must be positive")
    cell_width = args.cell_width or args.width
    if cell_width <= 0:
        raise ValueError("--width must be positive")
    if args.cols < 0:
        raise ValueError("--cols must be non-negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.speedup <= 0:
        raise ValueError("--speedup must be positive")
    if args.max_clips_per_video < 0:
        raise ValueError("--max-clips-per-video must be non-negative")

    rows, model_a_name, model_b_name = load_rows(args.csv, args.suite, args.winner, args.limit)
    if not rows:
        print("No rows matched the requested filters.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_dir / "grid_manifest.csv"
    report_md_path = args.report_md or args.output_dir / "report.md"
    report_html_path = args.report_html or args.output_dir / "report.html"
    manifest_records: list[dict[str, str | int]] = []

    if args.group_by_task:
        render_task_grouped_outputs(
            rows=rows,
            output_dir=args.output_dir,
            manifest_path=manifest_path,
            report_md_path=report_md_path,
            report_html_path=report_html_path,
            cell_width=cell_width,
            cols_arg=args.cols,
            fps=args.fps,
            speedup=args.speedup,
            max_clips_per_video=args.max_clips_per_video,
            ffmpeg=args.ffmpeg,
            crf=args.crf,
            preset=args.preset,
            keep_temp=args.keep_temp,
        )
        return 0

    total_pages = math.ceil(len(rows) / args.rows_per_page)
    for page_index in range(total_pages):
        start = page_index * args.rows_per_page
        page_rows = rows[start : start + args.rows_per_page]
        output_path = args.output_dir / page_name(page_index, page_rows)
        render_page(page_rows, output_path, model_a_name, model_b_name, cell_width, args.fps)
        print(f"Wrote {output_path} ({len(page_rows)} cases)")
        for page_row_index, row in enumerate(page_rows):
            manifest_records.append(
                {
                    "page": output_path.resolve(),
                    "page_index": page_index,
                    "row_in_page": page_row_index,
                    "csv_row": row.raw_index,
                    "suite": row.suite,
                    "seed": row.seed,
                    "horizon": row.horizon,
                    "task": row.task,
                    "episode": row.episode,
                    f"{model_a_name}_status": row.model_a_status,
                    f"{model_b_name}_status": row.model_b_status,
                    "winner": row.winner,
                    f"{model_a_name}_video": row.model_a_video,
                    f"{model_b_name}_video": row.model_b_video,
                }
            )

    write_manifest(manifest_path, manifest_records)
    print(f"Wrote manifest {manifest_path} ({len(manifest_records)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
