#!/bin/bash

#SBATCH --job-name=eval_fv_resub_l10
#SBATCH --output=slurm_output/eval_freeze_vlm_resubmit_libero10_%j.log
#SBATCH --error=slurm_output/eval_freeze_vlm_resubmit_libero10_%j.log
#SBATCH --time=72:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p logs slurm_output

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

EXP_NAME="freeze_vlm_resubmit"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
CKPT="${CKPT:-/fs/nexus-scratch/yliang17/Research/VLA/openpi/checkpoints_pi05_libero/pi05_libero/pi05_libero_20260510_210401/2000}"
PORT="${PORT:-8020}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
SEEDS="${SEEDS:-7 42 100}"
REPLAN_STEPS="${REPLAN_STEPS:-5 10}"
SERVER_READY_WAIT="${SERVER_READY_WAIT:-180}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXP_NAME}_libero10}"

mkdir -p "${RESULTS_DIR}" "data/${EXP_NAME}_libero10/videos"

python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${CKPT}" \
  > "logs/${EXP_NAME}_policy_server_libero10.log" 2>&1 &

SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

sleep "${SERVER_READY_WAIT}"

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for seed in ${SEEDS}; do
  for steps in ${REPLAN_STEPS}; do
    python examples/libero/main.py \
      --args.port "${PORT}" \
      --args.task-suite-name libero_10 \
      --args.seed "${seed}" \
      --args.replan-steps "${steps}" \
      --args.action-horizon "${steps}" \
      --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
      --args.results-dir "${RESULTS_DIR}" \
      --args.video-out-path "data/${EXP_NAME}_libero10/videos/seed${seed}_steps${steps}" \
      2>&1 | tee "logs/${EXP_NAME}_libero10_seed${seed}_steps${steps}.log"
  done
done
