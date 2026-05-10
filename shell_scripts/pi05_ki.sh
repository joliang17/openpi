

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

CHECKPOINT_BASE_DIR="/fs/nexus-projects/wilddiffusion/vla/openpi_ki"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

python scripts/compute_norm_stats.py  --config-name pi05_libero_ki_vlm_lora_action_expert

python3 scripts/train.py \
  pi05_libero_ki_vlm_lora_action_expert \
  --exp-name=pi05_ki_vlm_lora_action_expert_$(date +%Y%m%d_%H%M%S) \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"

