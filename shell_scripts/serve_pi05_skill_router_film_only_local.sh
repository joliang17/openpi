#!/bin/bash
# Run the pi0.5 skill-router film-only checkpoint as a local policy server.
set -euo pipefail

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p logs

source /etc/profile.d/modules.sh
module add cuda/12.8.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai
if [[ -f /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf ]]; then
  source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf
fi

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/fs/nexus-projects/wilddiffusion/vla/atomic_data}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/fs/nexus-projects/wilddiffusion/vla/openpi_skill_router}"
STAGE2_CONFIG="${STAGE2_CONFIG:-pi05_libero_skill_router_stage2_film_only}"
STAGE2_EXP="${STAGE2_EXP:-pi05_skill_router_stage2_film_only_20260511_191935}"
PORT="${PORT:-8010}"

STAGE2_DIR="${CHECKPOINT_BASE_DIR}/${STAGE2_CONFIG}/${STAGE2_EXP}"
if [[ ! -d "${STAGE2_DIR}" ]]; then
  echo "Checkpoint experiment directory not found: ${STAGE2_DIR}" >&2
  exit 1
fi

if [[ -n "${CKPT:-}" ]]; then
  POLICY_DIR="${CKPT}"
else
  STEP="${STEP:-$(ls "${STAGE2_DIR}" | grep -E '^[0-9]+$' | sort -n | tail -1)}"
  if [[ -z "${STEP}" ]]; then
    echo "No numeric checkpoint step found in ${STAGE2_DIR}" >&2
    exit 1
  fi
  POLICY_DIR="${STAGE2_DIR}/${STEP}"
fi

if [[ ! -d "${POLICY_DIR}/params" ]]; then
  echo "Checkpoint params directory not found: ${POLICY_DIR}/params" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Serving config=${STAGE2_CONFIG}"
echo "[$(date -Iseconds)] Checkpoint=${POLICY_DIR}"
echo "[$(date -Iseconds)] Port=${PORT}"

exec python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  --env LIBERO \
  policy:checkpoint \
  --policy.config="${STAGE2_CONFIG}" \
  --policy.dir="${POLICY_DIR}"
