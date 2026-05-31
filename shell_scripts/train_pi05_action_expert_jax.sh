#!/bin/bash

#SBATCH --job-name=openpi_finetune
#SBATCH --output=log/openpi_finetune.log
#SBATCH --error=log/openpi_finetune.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai

source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf

export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"

export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR
export HF_MODULES_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export OPENPI_DATA_HOME=$CACHE_DIR

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_LEROBOT_HOME="/fs/nexus-projects/wilddiffusion/vla/atomic_data"
export HF_HOME="/fs/nexus-projects/wilddiffusion/vla/libero_256"
export CUDA_VISIBLE_DEVICES=1


CONFIG_NAME="${CONFIG_NAME:-pi05_libero_action_expert}"
EXP_NAME="${EXP_NAME:-pi05_action_expert_jax_v2}"
RUN_NORM_STATS="${RUN_NORM_STATS:-1}"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# if [[ "${RUN_NORM_STATS}" == "1" ]]; then
# python3 scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
# fi

python3 scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" --resume
