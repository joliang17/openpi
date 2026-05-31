#!/bin/bash
# Evaluate a running local OpenPI policy server on LIBERO-10 and LIBERO-PRO.
set -euo pipefail

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p logs results data

source /etc/profile.d/modules.sh
module add cuda/12.8.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
if [[ -f /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf ]]; then
  source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf
fi

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/fs/nexus-projects/wilddiffusion/vla/atomic_data}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

HOST="${HOST:-localhost}"
PORT="${PORT:-8010}"
EXP_TAG="${EXP_TAG:-skill_router_lorabase_expert}"
SUITES="${SUITES:-libero10 libero_pro}"
SEEDS="${SEEDS:-7 42 100}"
# SEEDS="${SEEDS:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-5 10}"
NUM_TRIALS="${NUM_TRIALS:-10}"
PERTURB_TYPE="${PERTURB_TYPE:-object}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-30}"

wait_for_server() {
  python - "$HOST" "$PORT" "$SERVER_WAIT_SECONDS" <<'PY'
import socket
import sys
import time

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(1)
print(f"Server not reachable at {host}:{port} after {timeout:.0f}s", file=sys.stderr)
sys.exit(1)
PY
}

libero10_result_path() {
  local results_dir="$1" seed="$2" h="$3"
  printf '%s/libero_eval_modelopenpi_tasklibero_10_seed%s_h%s.json' "${results_dir}" "${seed}" "${h}"
}

run_libero10() {
  local results_dir="${RESULTS_LIB10:-results/${EXP_TAG}_libero10}"
  local video_dir="${VIDEOS_LIB10:-data/${EXP_TAG}_libero10/videos}"
  mkdir -p "${results_dir}" "${video_dir}"

  for seed in ${SEEDS}; do
    for h in ${REPLAN_STEPS}; do
      local out
      out="$(libero10_result_path "${results_dir}" "${seed}" "${h}")"
      if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}" ]]; then
        echo "[$(date -Iseconds)] skip libero10 seed=${seed} h=${h}: ${out}"
        continue
      fi

      echo "[$(date -Iseconds)] libero10 seed=${seed} h=${h}"
      python examples/libero/main.py \
        --args.host "${HOST}" \
        --args.port "${PORT}" \
        --args.task-suite-name libero_10 \
        --args.seed "${seed}" \
        --args.replan-steps "${h}" \
        --args.action-horizon "${h}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.results-dir "${results_dir}" \
        --args.video-out-path "${video_dir}/seed${seed}_h${h}" \
        2>&1 | tee "logs/${EXP_TAG}_libero10_seed${seed}_h${h}.log"
    done
  done
}

run_libero_pro() {
  local results_dir="${RESULTS_LIBPRO:-results/${EXP_TAG}_libero_pro}"
  local video_dir="${VIDEOS_LIBPRO:-data/${EXP_TAG}_libero_pro/videos}"
  mkdir -p "${results_dir}" "${video_dir}"

  for seed in ${SEEDS}; do
    for h in ${REPLAN_STEPS}; do
      local out
      out="$(libero10_result_path "${results_dir}" "${seed}" "${h}")"
      if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}" ]]; then
        echo "[$(date -Iseconds)] skip libero_pro seed=${seed} h=${h}: ${out}"
        continue
      fi

      echo "[$(date -Iseconds)] libero_pro seed=${seed} h=${h} perturb=${PERTURB_TYPE}"
      python examples/libero/main_pro.py \
        --host "${HOST}" \
        --port "${PORT}" \
        --task_suite_name libero_10 \
        --perturbation_type "${PERTURB_TYPE}" \
        --seed "${seed}" \
        --replan_steps "${h}" \
        --action_horizon "${h}" \
        --num_trials_per_task "${NUM_TRIALS}" \
        --results_dir "${results_dir}" \
        --video_out_path "${video_dir}/seed${seed}_h${h}" \
        2>&1 | tee "logs/${EXP_TAG}_libero_pro_seed${seed}_h${h}.log"
    done
  done
}

echo "[$(date -Iseconds)] Evaluating server ${HOST}:${PORT}"
echo "[$(date -Iseconds)] EXP_TAG=${EXP_TAG} SUITES=${SUITES} SEEDS=${SEEDS} REPLAN_STEPS=${REPLAN_STEPS}"
wait_for_server

for suite in ${SUITES}; do
  case "${suite}" in
    libero10)
      run_libero10
      ;;
    libero_pro)
      run_libero_pro
      ;;
    *)
      echo "Unknown suite '${suite}'. Use SUITES='libero10 libero_pro'." >&2
      exit 1
      ;;
  esac
done

echo "[$(date -Iseconds)] Eval done."
