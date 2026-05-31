#!/bin/bash
#SBATCH --job-name=pi05_vlm_lora_skill_router
#SBATCH --output=slurm_output/pi05_vlm_lora_skill_router_%j.log
#SBATCH --error=slurm_output/pi05_vlm_lora_skill_router_%j.log
#SBATCH --time=72:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G

# Two-stage skill-router finetuning on top of the LoRA-finetuned pi0.5 model.
#   Stage 1: trains the skill router (skill_pool_proj + skill_classifier).
#   Stage 2: trains the skill embedding + conditioning modules and the action expert.
# The LoRA-finetuned VLM weights are loaded and kept frozen in both stages.

set -euo pipefail

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi

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
export OPENPI_LIBERO_SKILL_ANNOTATION_PATH="${OPENPI_LIBERO_SKILL_ANNOTATION_PATH:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json}"

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

# LoRA-finetuned pi0.5 checkpoint that the skill router is built on top of.
export VLM_LORA_BASE_PARAMS="${VLM_LORA_BASE_PARAMS:-/fs/nexus-scratch/yliang17/Research/VLA/openpi/checkpoints_vlm_lora_action_expert/pi05_libero_vlm_lora_action_expert/pi05_vlm_lora_action_expert_20260508_155051/29999/params}"
if [[ ! -d "${VLM_LORA_BASE_PARAMS}" ]]; then
  echo "LoRA base params directory not found: ${VLM_LORA_BASE_PARAMS}" >&2
  exit 1
fi

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/fs/nexus-projects/wilddiffusion/vla/openpi_vlm_lora_skill_router}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

STAGE1_CONFIG="pi05_libero_vlm_lora_skill_router_stage1"
STAGE2_CONFIG="pi05_libero_vlm_lora_skill_router_stage2"
STAGE1_EXP_PROVIDED="${STAGE1_EXP:-}"
STAGE1_EXP="${STAGE1_EXP:-pi05_vlm_lora_skill_router_stage1_${RUN_TS}}"
STAGE2_EXP_PROVIDED="${STAGE2_EXP:-}"
STAGE2_EXP="${STAGE2_EXP:-pi05_vlm_lora_skill_router_stage2_from_${STAGE1_EXP}_${RUN_TS}}"

TRAIN_STAGE1="${TRAIN_STAGE1:-1}"
TRAIN_STAGE2="${TRAIN_STAGE2:-1}"
STAGE1_RESUME="${STAGE1_RESUME:-0}"
STAGE2_RESUME="${STAGE2_RESUME:-0}"
STAGE1_OVERWRITE="${STAGE1_OVERWRITE:-0}"
STAGE2_OVERWRITE="${STAGE2_OVERWRITE:-0}"
STAGE1_NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS:-30000}"
STAGE2_NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS:-30000}"
STAGE1_STEP="${STAGE1_STEP:-auto}"
STAGE1_PARAMS="${STAGE1_PARAMS:-}"

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

latest_checkpoint_step() {
  local checkpoint_dir="$1"
  ls "${checkpoint_dir}" | grep -E '^[0-9]+$' | sort -n | tail -1
}

if [[ "${TRAIN_STAGE1}" == "1" ]]; then
  compute_norm_stats_if_missing "${STAGE1_CONFIG}"

  STAGE1_ARGS=(
    "${STAGE1_CONFIG}"
    --exp-name="${STAGE1_EXP}"
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
    --num-train-steps="${STAGE1_NUM_TRAIN_STEPS}"
    --keep-period None
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

  echo "Stage 1 config: ${STAGE1_CONFIG}"
  echo "Stage 1 exp: ${STAGE1_EXP}"
  echo "Stage 1 base params: ${VLM_LORA_BASE_PARAMS}"
  echo "Stage 1 train steps: ${STAGE1_NUM_TRAIN_STEPS}"
  python3 scripts/train.py "${STAGE1_ARGS[@]}"

  STAGE1_DIR="${CHECKPOINT_BASE_DIR}/${STAGE1_CONFIG}/${STAGE1_EXP}"
  STAGE1_STEP="$(latest_checkpoint_step "${STAGE1_DIR}")"
  STAGE1_PARAMS="${STAGE1_DIR}/${STAGE1_STEP}/params"
else
  echo "TRAIN_STAGE1=${TRAIN_STAGE1}; skipping stage 1."
  if [[ -z "${STAGE1_PARAMS}" ]]; then
    if [[ "${STAGE1_STEP}" == "auto" ]]; then
      STAGE1_DIR="${CHECKPOINT_BASE_DIR}/${STAGE1_CONFIG}/${STAGE1_EXP}"
      STAGE1_STEP="$(latest_checkpoint_step "${STAGE1_DIR}")"
    fi
    STAGE1_PARAMS="${CHECKPOINT_BASE_DIR}/${STAGE1_CONFIG}/${STAGE1_EXP}/${STAGE1_STEP}/params"
  fi
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

STAGE2_ARGS=(
  "${STAGE2_CONFIG}"
  --exp-name="${STAGE2_EXP}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --num-train-steps="${STAGE2_NUM_TRAIN_STEPS}"
  --keep-period None
)

if [[ "${STAGE2_RESUME}" == "1" ]]; then
  if [[ -z "${STAGE2_EXP_PROVIDED}" ]]; then
    echo "STAGE2_RESUME=1 requires STAGE2_EXP to name an existing checkpoint folder." >&2
    exit 1
  fi
  STAGE2_ARGS+=(--resume)
elif [[ "${STAGE2_OVERWRITE}" == "1" ]]; then
  STAGE2_ARGS+=(--overwrite)
fi

echo "Stage 2 config: ${STAGE2_CONFIG}"
echo "Stage 2 exp: ${STAGE2_EXP}"
echo "Stage 2 train steps: ${STAGE2_NUM_TRAIN_STEPS}"
python3 scripts/train.py "${STAGE2_ARGS[@]}"
