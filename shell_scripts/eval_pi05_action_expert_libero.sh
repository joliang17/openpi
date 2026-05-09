#!/bin/bash

set -euo pipefail

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p logs data/pi05_action_expert_libero10/videos data/pi05_action_expert_libero_pro/videos

PORT="${PORT:-8005}"
CKPT="${CKPT:-/fs/nexus-scratch/yliang17/Research/VLA/openpi/checkpoints/pi05_libero_action_expert/pi05_action_expert_jax/29999}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
PERTURBATION_TYPE="${PERTURBATION_TYPE:-object}"
SEEDS="${SEEDS:-7 42 100}"
REPLAN_STEPS="${REPLAN_STEPS:-5 10}"
SERVER_READY_WAIT="${SERVER_READY_WAIT:-120}"
STATUS_DIR="${STATUS_DIR:-data/pi05_action_expert_eval_status}"

mkdir -p "${STATUS_DIR}"

json_value() {
  local value="${1:-}"
  if [[ -z "${value}" ]]; then
    printf 'null'
  else
    printf '%s' "${value}"
  fi
}

write_status_json() {
  local status_file="$1"
  local eval_type="$2"
  local seed="$3"
  local steps="$4"
  local status="$5"
  local log_path="$6"
  local video_out_path="$7"
  local exit_code="${8:-}"
  local success_rate="${9:-}"
  local success_percent="${10:-}"
  local started_at="${11:-}"
  local completed_at="${12:-}"

  cat > "${status_file}" <<EOF
{
  "eval_type": "${eval_type}",
  "task_suite": "libero_10",
  "perturbation_type": $(if [[ "${eval_type}" == "libero_pro" ]]; then printf '"%s"' "${PERTURBATION_TYPE}"; else printf 'null'; fi),
  "seed": ${seed},
  "replan_steps": ${steps},
  "num_trials_per_task": ${NUM_TRIALS_PER_TASK},
  "status": "${status}",
  "success_rate": $(json_value "${success_rate}"),
  "success_percent": $(json_value "${success_percent}"),
  "exit_code": $(json_value "${exit_code}"),
  "checkpoint": "${CKPT}",
  "log_path": "${log_path}",
  "video_out_path": "${video_out_path}",
  "started_at": $(if [[ -n "${started_at}" ]]; then printf '"%s"' "${started_at}"; else printf 'null'; fi),
  "completed_at": $(if [[ -n "${completed_at}" ]]; then printf '"%s"' "${completed_at}"; else printf 'null'; fi)
}
EOF
}

parse_libero10_success_rate() {
  local log_path="$1"
  grep "Total success rate:" "${log_path}" | tail -n 1 | awk '{print $NF}'
}

parse_libero_pro_success_percent() {
  local log_path="$1"
  grep "^Final:" "${log_path}" | tail -n 1 | sed -E 's/.* = ([0-9.]+)%.*/\1/'
}

initialize_status_files() {
  for seed in ${SEEDS}; do
    for steps in ${REPLAN_STEPS}; do
      write_status_json \
        "${STATUS_DIR}/libero10_seed${seed}_steps${steps}.json" \
        "libero10" \
        "${seed}" \
        "${steps}" \
        "pending" \
        "logs/pi05_action_expert_libero10_seed${seed}_steps${steps}.log" \
        "data/pi05_action_expert_libero10/videos/seed${seed}_steps${steps}"

      write_status_json \
        "${STATUS_DIR}/libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.json" \
        "libero_pro" \
        "${seed}" \
        "${steps}" \
        "pending" \
        "logs/pi05_action_expert_libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.log" \
        "data/pi05_action_expert_libero_pro/videos/seed${seed}_steps${steps}"
    done
  done
}

run_libero10_eval() {
  local seed="$1"
  local steps="$2"
  local log_path="logs/pi05_action_expert_libero10_seed${seed}_steps${steps}.log"
  local video_out_path="data/pi05_action_expert_libero10/videos/seed${seed}_steps${steps}"
  local status_file="${STATUS_DIR}/libero10_seed${seed}_steps${steps}.json"
  local started_at
  started_at="$(date -Iseconds)"

  write_status_json "${status_file}" "libero10" "${seed}" "${steps}" "running" "${log_path}" "${video_out_path}" "" "" "" "${started_at}"

  set +e
  python examples/libero/main.py \
    --args.port "${PORT}" \
    --args.task-suite-name libero_10 \
    --args.seed "${seed}" \
    --args.replan-steps "${steps}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  local completed_at success_rate success_percent status
  completed_at="$(date -Iseconds)"
  success_rate="$(parse_libero10_success_rate "${log_path}" || true)"
  success_percent="$(awk -v rate="${success_rate:-}" 'BEGIN { if (rate == "") print ""; else printf "%.2f", rate * 100 }')"
  status="finished"
  if [[ "${exit_code}" != "0" ]]; then
    status="failed"
  fi

  write_status_json "${status_file}" "libero10" "${seed}" "${steps}" "${status}" "${log_path}" "${video_out_path}" "${exit_code}" "${success_rate}" "${success_percent}" "${started_at}" "${completed_at}"
  return "${exit_code}"
}

run_libero_pro_eval() {
  local seed="$1"
  local steps="$2"
  local log_path="logs/pi05_action_expert_libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.log"
  local video_out_path="data/pi05_action_expert_libero_pro/videos/seed${seed}_steps${steps}"
  local status_file="${STATUS_DIR}/libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.json"
  local started_at
  started_at="$(date -Iseconds)"

  write_status_json "${status_file}" "libero_pro" "${seed}" "${steps}" "running" "${log_path}" "${video_out_path}" "" "" "" "${started_at}"

  set +e
  python examples/libero/main_pro.py \
    --task_suite_name libero_10 \
    --perturbation_type "${PERTURBATION_TYPE}" \
    --seed "${seed}" \
    --replan_steps "${steps}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --port "${PORT}" \
    --video_out_path "${video_out_path}" \
    2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  local completed_at success_rate success_percent status
  completed_at="$(date -Iseconds)"
  success_percent="$(parse_libero_pro_success_percent "${log_path}" || true)"
  success_rate="$(awk -v pct="${success_percent:-}" 'BEGIN { if (pct == "") print ""; else printf "%.4f", pct / 100 }')"
  status="finished"
  if [[ "${exit_code}" != "0" ]]; then
    status="failed"
  fi

  write_status_json "${status_file}" "libero_pro" "${seed}" "${steps}" "${status}" "${log_path}" "${video_out_path}" "${exit_code}" "${success_rate}" "${success_percent}" "${started_at}" "${completed_at}"
  return "${exit_code}"
}

initialize_status_files

python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config=pi05_libero_action_expert \
  --policy.dir="${CKPT}" \
  > logs/pi05_action_expert_policy_server.log 2>&1 &

SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep "${SERVER_READY_WAIT}"

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for SEED in ${SEEDS}; do
  for STEPS in ${REPLAN_STEPS}; do
    run_libero10_eval "${SEED}" "${STEPS}"
    run_libero_pro_eval "${SEED}" "${STEPS}"
  done
done
