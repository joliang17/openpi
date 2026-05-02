#!/bin/bash

#SBATCH --job-name=pi05
#SBATCH --output=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/pi05.log
#SBATCH --error=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/pi05.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai

# conda activate openai
export CACHE_DIR="/fs/nexus-scratch/yliang17/Research/cache"

export OPENPI_DATA_HOME=$CACHE_DIR
export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR
export HF_MODULES_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR

python3 download_model.py
python3 examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "${CACHE_DIR}/openpi-assets/checkpoints/pi05_base" \
    --config_name pi05_libero \
    --output_path "${CACHE_DIR}/openpi-assets/checkpoints/pi05_base_torch"

# export WANDB_PROJECT="vla_tooluse"

# # finetune
# # python3 scripts/compute_norm_stats.py --config-name=pi05_libero

# python3 scripts/train_pytorch.py pi05_libero --exp_name pi05_LIBERO_base --save_interval 10000