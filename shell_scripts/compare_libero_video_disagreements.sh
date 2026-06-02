#!/usr/bin/env bash
set -euo pipefail

# Compare two LIBERO eval video folders, then build task-grouped failure-only
# grid videos plus Markdown/HTML summary reports.
#
# Example:
#   bash shell_scripts/compare_libero_video_disagreements.sh \
#     pi05_lora_gated_film_skill_router_joint \
#     pi05_libero_lora_skill_router_joint
#
# Optional env overrides:
#   SUITE=both|libero10|libero_pro
#   DATA_ROOT=data
#   CSV_OUT=results_csv/my_compare.csv
#   OUT_DIR=data/video_grids/my_compare
#   COLS=4
#   CELL_WIDTH=180
#   FPS=10
#   SPEEDUP=2
#   LIMIT=0
#   FFMPEG=/opt/local/stow/ffmpeg-7.1/bin/ffmpeg
#   FORCE_EXTRACT=1   # rebuild CSV from success/failure videos; requires success videos to exist

# compare gated parameters
# MODEL_A="${1:-pi05_lora_gated_film_skill_router_joint}"
# MODEL_B="${2:-pi05_libero_lora_skill_router_joint}"

# compare best with fully finetune
MODEL_A="${1:-pi05_lora_gated_film_skill_router_joint}"
MODEL_B="${2:-pi05_libero}"

# SUITE="${SUITE:-both}"
SUITE="${SUITE:-libero10}"
DATA_ROOT="${DATA_ROOT:-data}"
COLS="${COLS:-4}"
CELL_WIDTH="${CELL_WIDTH:-180}"
FPS="${FPS:-20}"
SPEEDUP="${SPEEDUP:-2}"
LIMIT="${LIMIT:-0}"
FFMPEG="${FFMPEG:-/opt/local/stow/ffmpeg-7.1/bin/ffmpeg}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
# Python interpreter with the project deps (cv2/numpy/PIL). Plain `python` is not
# on PATH here, and a full `uv sync` fails on rerun-sdk, so skip the sync.
PYTHON="${PYTHON:-uv run --no-sync python}"

slugify() {
  ${PYTHON} - "$1" <<'PY'
import re
import sys
text = sys.argv[1]
print(re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower())
PY
}

COMPARE_NAME="${COMPARE_NAME:-$(slugify "${MODEL_A}_vs_${MODEL_B}")}"
CSV_OUT="${CSV_OUT:-results_csv/${COMPARE_NAME}_video_disagreements.csv}"
OUT_DIR="${OUT_DIR:-data/video_grids/${COMPARE_NAME}_task_failure_h264}"

echo "Model A: ${MODEL_A}"
echo "Model B: ${MODEL_B}"
echo "Suite: ${SUITE}"
echo "Data root: ${DATA_ROOT}"
echo "CSV: ${CSV_OUT}"
echo "Output dir: ${OUT_DIR}"
echo "Grid: cols=${COLS}, cell_width=${CELL_WIDTH}, fps=${FPS}, speedup=${SPEEDUP}"

if [[ "${FORCE_EXTRACT}" == "1" || ! -s "${CSV_OUT}" ]]; then
  echo "Extracting disagreement CSV from video folders..."
  echo "Note: this requires both success and failure source videos to exist."
  ${PYTHON} scripts/extract_video_disagreements.py \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --suite "${SUITE}" \
    --data-root "${DATA_ROOT}" \
    --limit "${LIMIT}" \
    --output "${CSV_OUT}" \
    > /tmp/"${COMPARE_NAME}_video_disagreements_preview.csv"
else
  echo "Using existing disagreement CSV: ${CSV_OUT}"
  echo "Set FORCE_EXTRACT=1 to rebuild it from source videos."
fi

${PYTHON} scripts/make_video_disagreement_grid.py \
  --csv "${CSV_OUT}" \
  --group-by-task \
  --cell-width "${CELL_WIDTH}" \
  --cols "${COLS}" \
  --fps "${FPS}" \
  --speedup "${SPEEDUP}" \
  --ffmpeg "${FFMPEG}" \
  --limit "${LIMIT}" \
  --output-dir "${OUT_DIR}"

echo
echo "Done."
echo "CSV: ${CSV_OUT}"
echo "Grid videos: ${OUT_DIR}"
echo "Markdown report: ${OUT_DIR}/report.md"
echo "HTML report: ${OUT_DIR}/report.html"
echo "Manifest: ${OUT_DIR}/grid_manifest.csv"
