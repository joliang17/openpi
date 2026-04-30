#!/bin/bash
#SBATCH --job-name=atmoicVLA2
#SBATCH --output=slurm_output/atmoicVLA2.log
#SBATCH --error=slurm_output/atmoicVLA2.log
#SBATCH --time=72:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h100-nvl:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G

source /etc/profile.d/modules.sh
module add cuda/12.8.1
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

export WANDB_MODE=disabled

# ---- JAX training (original) ----
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ---- PyTorch training ----
export NCCL_DEBUG=WARN

# python3 scripts/compute_norm_stats.py --config-name Atomic_libero_pytorch



python3 -u scripts/serve_policy.py \
  --port=8005 > "pi05.log" 2>&1 &


