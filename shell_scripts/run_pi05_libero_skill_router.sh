#!/bin/bash
#SBATCH --job-name=pi05_skill_router
#SBATCH --output=slurm_output/pi05_skill_router.log
#SBATCH --error=slurm_output/pi05_skill_router.log
#SBATCH --time=72:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G

set -euo pipefail

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi

export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR"
export HF_MODULES_CACHE="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export OPENPI_DATA_HOME="$CACHE_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export HF_LEROBOT_HOME="/fs/nexus-projects/wilddiffusion/vla/atomic_data"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OPENPI_LIBERO_SKILL_ANNOTATION_PATH="${OPENPI_LIBERO_SKILL_ANNOTATION_PATH:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10_half.json}"

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

CHECKPOINT_BASE_DIR="/fs/nexus-projects/wilddiffusion/vla/openpi_skill_router"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
STAGE1_EXP_PROVIDED="${STAGE1_EXP:-}"
STAGE1_EXP="${STAGE1_EXP:-pi05_skill_router_stage1_20260507_101717}"
STAGE2_EXP_PROVIDED="${STAGE2_EXP:-}"
STAGE2_EXP="${STAGE2_EXP:-pi05_skill_router_stage2_from_stage1_20260507_101717_step6000_${RUN_TS}}"
STAGE1_STEP="${STAGE1_STEP:-6000}"
TRAIN_STAGE1="${TRAIN_STAGE1:-0}"
STAGE1_RESUME="${STAGE1_RESUME:-0}"
STAGE1_OVERWRITE="${STAGE1_OVERWRITE:-0}"
STAGE1_NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS:-60000}"
STAGE2_RESUME="${STAGE2_RESUME:-0}"
STAGE2_NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS:-30000}"
TRAIN_STAGE2="${TRAIN_STAGE2:-1}"
STAGE1_PARAMS="${STAGE1_PARAMS:-${CHECKPOINT_BASE_DIR}/pi05_libero_skill_router_stage1/${STAGE1_EXP}/${STAGE1_STEP}/params}"

compute_norm_stats_if_missing() {
  local config_name="$1"
  local stats_file="assets/${config_name}/libero_atomic/norm_stats.json"

  if [[ -f "${stats_file}" ]]; then
    echo "Found norm stats for ${config_name}: ${stats_file}"
    return
  fi

  echo "Computing norm stats for ${config_name}"
  python3 scripts/compute_norm_stats.py --config-name "${config_name}"
}

STAGE1_CONFIG="pi05_libero_skill_router_stage1"
STAGE2_CONFIG="pi05_libero_skill_router_stage2"

if [[ "${TRAIN_STAGE1}" == "1" ]]; then
  compute_norm_stats_if_missing "${STAGE1_CONFIG}"

  STAGE1_ARGS=(
    "${STAGE1_CONFIG}"
    --exp-name="${STAGE1_EXP}"
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
    --num-train-steps="${STAGE1_NUM_TRAIN_STEPS}"
  )

  if [[ "${STAGE1_RESUME}" == "1" ]]; then
    if [[ -z "${STAGE1_EXP_PROVIDED}" ]]; then
      echo "STAGE1_RESUME=1 requires STAGE1_EXP to name an existing checkpoint folder." >&2
      exit 1
    fi
    STAGE1_ARGS+=(--resume)
  elif [[ "${STAGE1_OVERWRITE}" == "1" ]]; then
    STAGE1_ARGS+=(--overwrite)
  fi

  echo "Stage 1 checkpoint exp: ${STAGE1_EXP}"
  echo "Stage 1 resume: ${STAGE1_RESUME}"
  echo "Stage 1 train steps: ${STAGE1_NUM_TRAIN_STEPS}"
  python3 scripts/train.py "${STAGE1_ARGS[@]}"

  STAGE1_STEP_FOR_STAGE2="${STAGE1_STEP}"
  if [[ -z "${STAGE1_STEP_FOR_STAGE2}" || "${STAGE1_STEP_FOR_STAGE2}" == "auto" ]]; then
    STAGE1_STEP_FOR_STAGE2="$((STAGE1_NUM_TRAIN_STEPS - 1))"
  fi
  STAGE1_PARAMS="${CHECKPOINT_BASE_DIR}/pi05_libero_skill_router_stage1/${STAGE1_EXP}/${STAGE1_STEP_FOR_STAGE2}/params"
else
  echo "TRAIN_STAGE1=${TRAIN_STAGE1}; skipping stage 1."
fi

if [[ ! -d "${STAGE1_PARAMS}" ]]; then
  echo "Stage 1 params directory not found: ${STAGE1_PARAMS}" >&2
  exit 1
fi
export OPENPI_SKILL_STAGE1_PARAMS="${STAGE1_PARAMS}"
echo "Stage 1 params for stage 2: ${OPENPI_SKILL_STAGE1_PARAMS}"

if [[ "${TRAIN_STAGE2}" != "1" ]]; then
  echo "TRAIN_STAGE2=${TRAIN_STAGE2}; skipping stage 2."
  exit 0
fi

compute_norm_stats_if_missing "${STAGE2_CONFIG}"

TRAIN_ARGS=(
  "${STAGE2_CONFIG}"
  --exp-name="${STAGE2_EXP}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --num-train-steps="${STAGE2_NUM_TRAIN_STEPS}"
)

if [[ "${STAGE2_RESUME}" == "1" ]]; then
  if [[ -z "${STAGE2_EXP_PROVIDED}" ]]; then
    echo "STAGE2_RESUME=1 requires STAGE2_EXP to name an existing checkpoint folder." >&2
    exit 1
  fi
  TRAIN_ARGS+=(--resume)
fi

echo "Stage 2 checkpoint exp: ${STAGE2_EXP}"
echo "Stage 2 resume: ${STAGE2_RESUME}"
echo "Stage 2 train steps: ${STAGE2_NUM_TRAIN_STEPS}"
python3 scripts/train.py "${TRAIN_ARGS[@]}"
