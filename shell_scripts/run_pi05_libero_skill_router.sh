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
STAGE1_EXP="pi05_skill_router_stage1"
STAGE2_EXP="pi05_skill_router_stage2"
STAGE1_STEP="${STAGE1_STEP:-29999}"

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

compute_norm_stats_if_missing "${STAGE1_CONFIG}"

python3 scripts/train.py "${STAGE1_CONFIG}" \
  --exp-name="${STAGE1_EXP}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --resume

export OPENPI_SKILL_STAGE1_PARAMS="${CHECKPOINT_BASE_DIR}/pi05_libero_skill_router_stage1/${STAGE1_EXP}/${STAGE1_STEP}/params"

compute_norm_stats_if_missing "${STAGE2_CONFIG}"

python3 scripts/train.py "${STAGE2_CONFIG}" \
  --exp-name="${STAGE2_EXP}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --resume
