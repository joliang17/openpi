#!/bin/bash

#SBATCH --job-name=eval_ki_libpro
#SBATCH --output=slurm_output/eval_ki_libero_pro.log
#SBATCH --error=slurm_output/eval_ki_libero_pro.log
#SBATCH --time=72:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

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

EXP_NAME="ki"
POLICY_CONFIG="pi05_libero_ki_vlm_lora_action_expert"
PORT="${PORT:-8007}"
CKPT="${CKPT:-/fs/nexus-projects/wilddiffusion/vla/openpi_ki/pi05_libero_ki_vlm_lora_action_expert/pi05_ki_vlm_lora_action_expert_20260509_120950/29999}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
PERTURBATION_TYPE="${PERTURBATION_TYPE:-object}"
SEEDS="${SEEDS:-7 42 100}"
REPLAN_STEPS="${REPLAN_STEPS:-5 10}"
SERVER_READY_WAIT="${SERVER_READY_WAIT:-180}"
STATUS_DIR="${STATUS_DIR:-data/${EXP_NAME}_libero_pro_eval_status}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXP_NAME}_libero_pro}"
SKIP_COMPLETED_RESULTS="${SKIP_COMPLETED_RESULTS:-1}"

mkdir -p "${STATUS_DIR}" "${RESULTS_DIR}" "data/${EXP_NAME}_libero_pro/videos" logs

json_value() {
  local value="${1:-}"
  if [[ -z "${value}" ]]; then printf 'null'; else printf '%s' "${value}"; fi
}

write_status_json() {
  local status_file="$1" eval_type="$2" seed="$3" steps="$4" status="$5"
  local log_path="$6" video_out_path="$7" exit_code="${8:-}"
  local success_rate="${9:-}" success_percent="${10:-}"
  local started_at="${11:-}" completed_at="${12:-}"
  cat > "${status_file}" <<EOF
{
  "eval_type": "${eval_type}",
  "task_suite": "libero_10",
  "perturbation_type": "${PERTURBATION_TYPE}",
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

parse_libero_pro_success_percent() {
  grep "^Final:" "$1" | tail -n 1 | sed -E 's/.* = ([0-9.]+)%.*/\1/'
}

result_path_for() {
  printf '%s/libero_pro_eval_modelopenpi_tasklibero_10_%s_seed%s_h%s.json' \
    "${RESULTS_DIR}" "${PERTURBATION_TYPE}" "$1" "$2"
}

run_libero_pro_eval() {
  local seed="$1" steps="$2"
  local log_path="logs/${EXP_NAME}_libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.log"
  local video_out_path="data/${EXP_NAME}_libero_pro/videos/seed${seed}_steps${steps}"
  local status_file="${STATUS_DIR}/libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.json"
  local result_path; result_path="$(result_path_for "${seed}" "${steps}")"
  local started_at; started_at="$(date -Iseconds)"

  if [[ "${SKIP_COMPLETED_RESULTS}" == "1" && -s "${result_path}" ]]; then
    echo "Skipping libero_pro seed=${seed} steps=${steps}; found ${result_path}"
    write_status_json "${status_file}" "libero_pro" "${seed}" "${steps}" "skipped" \
      "${log_path}" "${video_out_path}" "0" "" "" "${started_at}" "$(date -Iseconds)"
    return 0
  fi

  write_status_json "${status_file}" "libero_pro" "${seed}" "${steps}" "running" \
    "${log_path}" "${video_out_path}" "" "" "" "${started_at}"

  set +e
  python examples/libero/main_pro.py \
    --task_suite_name libero_10 \
    --perturbation_type "${PERTURBATION_TYPE}" \
    --seed "${seed}" \
    --replan_steps "${steps}" \
    --action_horizon "${steps}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --results_dir "${RESULTS_DIR}" \
    --port "${PORT}" \
    --video_out_path "${video_out_path}" \
    2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  local completed_at; completed_at="$(date -Iseconds)"
  local success_percent; success_percent="$(parse_libero_pro_success_percent "${log_path}" || true)"
  local success_rate; success_rate="$(awk -v p="${success_percent:-}" 'BEGIN { if (p=="") print ""; else printf "%.4f", p/100 }')"
  local status="finished"
  [[ "${exit_code}" != "0" ]] && status="failed"

  write_status_json "${status_file}" "libero_pro" "${seed}" "${steps}" "${status}" \
    "${log_path}" "${video_out_path}" "${exit_code}" "${success_rate}" "${success_percent}" \
    "${started_at}" "${completed_at}"
  return "${exit_code}"
}

for seed in ${SEEDS}; do
  for steps in ${REPLAN_STEPS}; do
    write_status_json \
      "${STATUS_DIR}/libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.json" \
      "libero_pro" "${seed}" "${steps}" "pending" \
      "logs/${EXP_NAME}_libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${steps}.log" \
      "data/${EXP_NAME}_libero_pro/videos/seed${seed}_steps${steps}"
  done
done

python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${CKPT}" \
  > "logs/${EXP_NAME}_policy_server_libero_pro.log" 2>&1 &

SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

sleep "${SERVER_READY_WAIT}"

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for SEED in ${SEEDS}; do
  for STEPS in ${REPLAN_STEPS}; do
    run_libero_pro_eval "${SEED}" "${STEPS}"
  done
done
