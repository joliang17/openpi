#!/bin/bash
#SBATCH --job-name=pi05_vlm_lora_sr_joint
#SBATCH --output=slurm_output/pi05_vlm_lora_skill_router_joint_%j.log
#SBATCH --error=slurm_output/pi05_vlm_lora_skill_router_joint_%j.log
#SBATCH --time=72:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G

set -euo pipefail

source /etc/profile.d/modules.sh
module add cuda/12.8.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai

if [[ -f /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf ]]; then
  source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf
fi

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p slurm_output logs

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/fs/nexus-projects/wilddiffusion/vla/atomic_data}"
export OPENPI_LIBERO_SKILL_ANNOTATION_PATH="${OPENPI_LIBERO_SKILL_ANNOTATION_PATH:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero_vlm_lora_skill_router_joint}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-pi05_vlm_lora_skill_router_joint_${RUN_TS}}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/fs/nexus-projects/wilddiffusion/vla/openpi_vlm_lora_skill_router}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
RUN_NORM_STATS="${RUN_NORM_STATS:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"

if [[ "${RUN_NORM_STATS}" == "1" ]]; then
  python3 scripts/compute_norm_stats.py --config-name="${CONFIG_NAME}"
fi

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --num-train-steps="${NUM_TRAIN_STEPS}"
  --keep-period None
)

if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
elif [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi

echo "[$(date -Iseconds)] JOB=${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Checkpoint base: ${CHECKPOINT_BASE_DIR}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Skill annotation: ${OPENPI_LIBERO_SKILL_ANNOTATION_PATH}"
python3 scripts/train.py "${TRAIN_ARGS[@]}"
echo "[$(date -Iseconds)] Training done: ${CONFIG_NAME}/${EXP_NAME}"
