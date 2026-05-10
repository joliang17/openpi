#!/bin/bash

#SBATCH --job-name=eval_lora_lib10
#SBATCH --output=slurm_output/eval_vlm_lora_libero10.log
#SBATCH --error=slurm_output/eval_vlm_lora_libero10.log
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

EXP_NAME="vlm_lora"
POLICY_CONFIG="pi05_libero_vlm_lora_action_expert"
PORT="${PORT:-8008}"
CKPT="${CKPT:-/fs/nexus-scratch/yliang17/Research/VLA/openpi/checkpoints_vlm_lora_action_expert/pi05_libero_vlm_lora_action_expert/pi05_vlm_lora_action_expert_20260508_155051/29999}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
SEEDS="${SEEDS:-7 42 100}"
REPLAN_STEPS="${REPLAN_STEPS:-5 10}"
SERVER_READY_WAIT="${SERVER_READY_WAIT:-180}"
STATUS_DIR="${STATUS_DIR:-data/${EXP_NAME}_libero10_eval_status}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXP_NAME}_libero10}"
SKIP_COMPLETED_RESULTS="${SKIP_COMPLETED_RESULTS:-1}"

mkdir -p "${STATUS_DIR}" "${RESULTS_DIR}" "data/${EXP_NAME}_libero10/videos" logs

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
  "perturbation_type": null,
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
  grep "Total success rate:" "$1" | tail -n 1 | awk '{print $NF}'
}

result_path_for() {
  printf '%s/libero_eval_modelopenpi_tasklibero_10_seed%s_h%s.json' "${RESULTS_DIR}" "$1" "$2"
}

run_libero10_eval() {
  local seed="$1" steps="$2"
  local log_path="logs/${EXP_NAME}_libero10_seed${seed}_steps${steps}.log"
  local video_out_path="data/${EXP_NAME}_libero10/videos/seed${seed}_steps${steps}"
  local status_file="${STATUS_DIR}/libero10_seed${seed}_steps${steps}.json"
  local result_path; result_path="$(result_path_for "${seed}" "${steps}")"
  local started_at; started_at="$(date -Iseconds)"

  if [[ "${SKIP_COMPLETED_RESULTS}" == "1" && -s "${result_path}" ]]; then
    echo "Skipping libero10 seed=${seed} steps=${steps}; found ${result_path}"
    write_status_json "${status_file}" "libero10" "${seed}" "${steps}" "skipped" \
      "${log_path}" "${video_out_path}" "0" "" "" "${started_at}" "$(date -Iseconds)"
    return 0
  fi

  write_status_json "${status_file}" "libero10" "${seed}" "${steps}" "running" \
    "${log_path}" "${video_out_path}" "" "" "" "${started_at}"

  set +e
  python examples/libero/main.py \
    --args.port "${PORT}" \
    --args.task-suite-name libero_10 \
    --args.seed "${seed}" \
    --args.replan-steps "${steps}" \
    --args.action-horizon "${steps}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.results-dir "${RESULTS_DIR}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  local completed_at; completed_at="$(date -Iseconds)"
  local success_rate; success_rate="$(parse_libero10_success_rate "${log_path}" || true)"
  local success_percent; success_percent="$(awk -v r="${success_rate:-}" 'BEGIN { if (r=="") print ""; else printf "%.2f", r*100 }')"
  local status="finished"
  [[ "${exit_code}" != "0" ]] && status="failed"

  write_status_json "${status_file}" "libero10" "${seed}" "${steps}" "${status}" \
    "${log_path}" "${video_out_path}" "${exit_code}" "${success_rate}" "${success_percent}" \
    "${started_at}" "${completed_at}"
  return "${exit_code}"
}

for seed in ${SEEDS}; do
  for steps in ${REPLAN_STEPS}; do
    write_status_json \
      "${STATUS_DIR}/libero10_seed${seed}_steps${steps}.json" \
      "libero10" "${seed}" "${steps}" "pending" \
      "logs/${EXP_NAME}_libero10_seed${seed}_steps${steps}.log" \
      "data/${EXP_NAME}_libero10/videos/seed${seed}_steps${steps}"
  done
done

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

for SEED in ${SEEDS}; do
  for STEPS in ${REPLAN_STEPS}; do
    run_libero10_eval "${SEED}" "${STEPS}"
  done
done
