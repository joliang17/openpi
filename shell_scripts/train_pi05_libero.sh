#!/bin/bash
#SBATCH --job-name=pi05_libero
#SBATCH --output=slurm_output/pi05_libero_%j.log
#SBATCH --error=slurm_output/pi05_libero_%j.log
#SBATCH --time=72:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G

set -euo pipefail

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p slurm_output

source /etc/profile.d/modules.sh
module add cuda/12.8.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai
source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/fs/nexus-projects/wilddiffusion/vla/atomic_data}"
export HF_HOME="${HF_HOME:-/fs/nexus-projects/wilddiffusion/vla/libero_256}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

RUN_TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME="${EXP_NAME:-pi05_libero_${RUN_TS}}"
CKPT_DIR="${CKPT_DIR:-/fs/nexus-projects/wilddiffusion/vla/openpi/checkpoints_pi05_libero}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${CKPT_DIR}" logs

TRAIN_ARGS=(
  pi05_libero
  --exp-name="${EXP_NAME}"
  --checkpoint-base-dir="${CKPT_DIR}"
  --batch-size="${BATCH_SIZE:-16}"
  --keep-period None
)
if [[ "${RESUME}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
elif [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi

python3 scripts/train.py "${TRAIN_ARGS[@]}"
