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

mkdir -p logs slurm_output

EVAL_POLICY_MODE="${EVAL_POLICY_MODE:-${1:-pi05}}"
POLICY_PORT="${POLICY_PORT:-8005}"

PI05_POLICY_CONFIG="${PI05_POLICY_CONFIG:-pi05_libero}"
PI05_POLICY_DIR="${PI05_POLICY_DIR:-gs://openpi-assets/checkpoints/pi05_libero}"

RUN_NAME="pi05"

# echo "Evaluation policy mode: ${EVAL_POLICY_MODE}"
# echo "Policy log: ${POLICY_LOG}"

# SERVER_PID=""
# cleanup() {
#   if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
#     kill "${SERVER_PID}"
#     wait "${SERVER_PID}" 2>/dev/null || true
#   fi
# }
# trap cleanup EXIT

# # ---- start policy server ----
# python3 -u scripts/serve_policy.py "${SERVER_ARGS[@]}" > "${POLICY_LOG}" 2>&1 &
# SERVER_PID=$!
# echo "Server PID: $SERVER_PID"

# # wait until server is ready
# echo "Waiting for server to be ready..."
# for i in {1..120}; do
#   if curl -sf "http://127.0.0.1:${POLICY_PORT}/healthz" > /dev/null 2>&1; then
#     echo "Server is ready."
#     break
#   fi

#   if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
#     echo "Server crashed. Last lines of ${POLICY_LOG}:"
#     tail -n 100 "${POLICY_LOG}"
#     exit 1
#   fi

#   sleep 5
# done

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ---- sweep config ----
SEEDS=(7 42 100)
REPLAN_STEPS=(5 10)


# ---- libero_10 (standard benchmark) ----
for SEED in "${SEEDS[@]}"; do
  for STEPS in "${REPLAN_STEPS[@]}"; do
    echo "=== libero_10  seed=${SEED}  replan_steps=${STEPS} ==="
    python examples/libero/main.py \
      --args.port ${POLICY_PORT} \
      --args.task-suite-name libero_10 \
      --args.seed ${SEED} \
      --args.replan-steps ${STEPS} \
      --args.num-trials-per-task 10 \
      --args.video-out-path "data/${RUN_NAME}_libero10/videos/seed${SEED}_steps${STEPS}" \
      2>&1 | tee "logs/${RUN_NAME}_libero10_seed${SEED}_steps${STEPS}.log"
  done
done


# ---- libero_pro (perturbation_type=none by default) ----
for SEED in "${SEEDS[@]}"; do
  for STEPS in "${REPLAN_STEPS[@]}"; do
    echo "=== libero_pro  seed=${SEED}  replan_steps=${STEPS} ==="
    python examples/libero/main_pro.py \
      --task_suite_name libero_10 \
      --perturbation_type object \
      --seed ${SEED} \
      --replan_steps ${STEPS} \
      --num_trials_per_task 10 \
      --port "${POLICY_PORT}" \
      --video_out_path "data/${RUN_NAME}_libero_pro/videos/seed${SEED}_steps${STEPS}" \
      2>&1 | tee "logs/${RUN_NAME}_libero_pro_seed${SEED}_steps${STEPS}.log"
  done
done

