#!/bin/bash
#SBATCH --job-name=pi05_sr_ablation_eval
#SBATCH --output=slurm_output/pi05_sr_ablation_eval_%j.log
#SBATCH --error=slurm_output/pi05_sr_ablation_eval_%j.log
#SBATCH --time=12:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# Eval for skill-router stage2 ablations.
# Driven by VARIANT={adarms_only|film_only|film_vlm} and SUITE={libero10|libero_pro}.
# Submit (example):
#   sbatch --export=ALL,VARIANT=film_only,SUITE=libero10 shell_scripts/sbatch_pi05_skill_router_ablation_eval.sh
#   sbatch --export=ALL,VARIANT=film_only,SUITE=libero_pro shell_scripts/sbatch_pi05_skill_router_ablation_eval.sh

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
export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/fs/nexus-projects/wilddiffusion/vla/atomic_data}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

############################################
# Variant / suite config
############################################
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/fs/nexus-projects/wilddiffusion/vla/openpi_skill_router}"

case "${VARIANT:?set VARIANT=adarms_only|film_only|film_vlm}" in
  adarms_only)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_adarms_only
    STAGE2_EXP="${STAGE2_EXP:-pi05_skill_router_stage2_adarms_only_20260511_191930}"
    EXP_TAG_DEFAULT=skill_router_adarms_only
    ;;
  film_only)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_film_only
    STAGE2_EXP="${STAGE2_EXP:-pi05_skill_router_stage2_film_only_20260511_191935}"
    EXP_TAG_DEFAULT=skill_router_film_only
    ;;
  film_vlm)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_film_vlm
    STAGE2_EXP="${STAGE2_EXP:-pi05_skill_router_stage2_film_vlm_20260511_191935}"
    EXP_TAG_DEFAULT=skill_router_film_vlm
    ;;
  *)
    echo "unknown VARIANT=${VARIANT}" >&2; exit 1 ;;
esac

STAGE2_DIR="${CHECKPOINT_BASE_DIR}/${STAGE2_CONFIG}/${STAGE2_EXP}"
LATEST_STEP=$(ls "${STAGE2_DIR}" | grep -E '^[0-9]+$' | sort -n | tail -1)
CKPT="${STAGE2_DIR}/${LATEST_STEP}"
EXP_TAG="${EXP_TAG:-${EXP_TAG_DEFAULT}}"
PORT="${PORT:-8010}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SEEDS="${SEEDS:-7 42 100}"
REPLAN_STEPS="${REPLAN_STEPS:-5 10}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SERVER_READY_WAIT="${SERVER_READY_WAIT:-360}"

echo "[$(date -Iseconds)] JOB=${SLURM_JOB_ID:-local} VARIANT=${VARIANT} SUITE=${SUITE:?set SUITE=libero10|libero_pro} CKPT=${CKPT}"

############################################
# Policy server (openai env)
############################################
python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  --env libero \
  policy:checkpoint \
  --policy.config="${STAGE2_CONFIG}" \
  --policy.dir="${CKPT}" \
  > "logs/${EXP_TAG}_${SUITE}_policy_server.log" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} >/dev/null 2>&1 || true" EXIT
echo "[$(date -Iseconds)] Policy server PID=${SERVER_PID}; waiting ${SERVER_READY_WAIT}s"
sleep "${SERVER_READY_WAIT}"

############################################
# Eval client (gr00t env)
############################################
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

result_path() { printf '%s/libero_eval_modelopenpi_tasklibero_10_seed%s_h%s.json' "$1" "$2" "$3"; }

if [[ "${SUITE}" == "libero10" ]]; then
  RESULTS_DIR="${RESULTS_DIR:-results/${EXP_TAG}_libero10}"
  mkdir -p "${RESULTS_DIR}" "data/${EXP_TAG}_libero10/videos"
  for seed in ${SEEDS}; do
    for h in ${REPLAN_STEPS}; do
      out=$(result_path "${RESULTS_DIR}" "${seed}" "${h}")
      if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}" ]]; then
        echo "[$(date -Iseconds)] skip seed=${seed} h=${h}"; continue
      fi
      echo "[$(date -Iseconds)] libero10 seed=${seed} h=${h}"
      python examples/libero/main.py \
        --args.port "${PORT}" \
        --args.task-suite-name libero_10 \
        --args.seed "${seed}" \
        --args.replan-steps "${h}" \
        --args.action-horizon "${h}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.results-dir "${RESULTS_DIR}" \
        --args.video-out-path "data/${EXP_TAG}_libero10/videos/seed${seed}_h${h}" \
        2>&1 | tee "logs/${EXP_TAG}_libero10_seed${seed}_h${h}.log"
    done
  done

elif [[ "${SUITE}" == "libero_pro" ]]; then
  RESULTS_DIR="${RESULTS_DIR:-results/${EXP_TAG}_libero_pro}"
  PERTURB_TYPE="${PERTURB_TYPE:-object}"
  mkdir -p "${RESULTS_DIR}" "data/${EXP_TAG}_libero_pro/videos"
  for seed in ${SEEDS}; do
    for h in ${REPLAN_STEPS}; do
      out=$(result_path "${RESULTS_DIR}" "${seed}" "${h}")
      if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}" ]]; then
        echo "[$(date -Iseconds)] skip seed=${seed} h=${h}"; continue
      fi
      echo "[$(date -Iseconds)] libero_pro seed=${seed} h=${h}"
      python examples/libero/main_pro.py \
        --task_suite_name libero_10 \
        --perturbation_type "${PERTURB_TYPE}" \
        --seed "${seed}" \
        --replan_steps "${h}" \
        --action_horizon "${h}" \
        --num_trials_per_task "${NUM_TRIALS}" \
        --results_dir "${RESULTS_DIR}" \
        --port "${PORT}" \
        --video_out_path "data/${EXP_TAG}_libero_pro/videos/seed${seed}_h${h}" \
        2>&1 | tee "logs/${EXP_TAG}_libero_pro_seed${seed}_h${h}.log"
    done
  done

else
  echo "unknown SUITE=${SUITE}" >&2; exit 1
fi

echo "[$(date -Iseconds)] === Eval done (variant=${VARIANT} suite=${SUITE}) ==="
