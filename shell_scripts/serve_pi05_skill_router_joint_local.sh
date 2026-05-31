#!/bin/bash
# Run the pi0.5 VLM-LoRA skill-router joint checkpoint as a local policy server.
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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/fs/nexus-projects/wilddiffusion/vla/atomic_data}"
export HF_HOME="${HF_HOME:-/fs/nexus-projects/wilddiffusion/vla/libero_256}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_vlm_lora_skill_router_joint}"
CKPT="${CKPT:-/fs/nexus-projects/wilddiffusion/vla/openpi_vlm_lora_skill_router/pi05_libero_vlm_lora_skill_router_joint/pi05_vlm_lora_skill_router_joint_20260515_113425/29999}"
PORT="${PORT:-8025}"

if [[ ! -d "${CKPT}/params" ]]; then
  echo "Checkpoint params directory not found: ${CKPT}/params" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Serving config=${POLICY_CONFIG}"
echo "[$(date -Iseconds)] Checkpoint=${CKPT}"
echo "[$(date -Iseconds)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[$(date -Iseconds)] Port=${PORT}"

exec python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  --env LIBERO \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${CKPT}"
