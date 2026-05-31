#!/bin/bash
# Skill-router stage2 ablations: adarms_only | film_only | film_vlm
# Driven by VARIANT env var.
set -euo pipefail

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p logs results

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
export OPENPI_LIBERO_SKILL_ANNOTATION_PATH="${OPENPI_LIBERO_SKILL_ANNOTATION_PATH:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10_half.json}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

############################################
# Variant selection
############################################
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
case "${VARIANT:?set VARIANT=adarms_only|film_only|film_vlm}" in
  adarms_only)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_adarms_only
    STAGE2_EXP_DEFAULT=pi05_skill_router_stage2_adarms_only_${RUN_TAG}
    EXP_TAG_DEFAULT=skill_router_adarms_only
    ;;
  film_only)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_film_only
    STAGE2_EXP_DEFAULT=pi05_skill_router_stage2_film_only_${RUN_TAG}
    EXP_TAG_DEFAULT=skill_router_film_only
    ;;
  film_vlm)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_film_vlm
    STAGE2_EXP_DEFAULT=pi05_skill_router_stage2_film_vlm_${RUN_TAG}
    EXP_TAG_DEFAULT=skill_router_film_vlm
    ;;
  *)
    echo "unknown VARIANT=${VARIANT}" >&2
    exit 1
    ;;
esac

CHECKPOINT_BASE_DIR="/fs/nexus-projects/wilddiffusion/vla/openpi_skill_router"
STAGE1_EXP="${STAGE1_EXP:-pi05_skill_router_stage1_20260507_101717}"
STAGE1_STEP="${STAGE1_STEP:-6000}"
STAGE2_EXP="${STAGE2_EXP:-${STAGE2_EXP_DEFAULT}}"
STAGE2_NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS:-30000}"
STAGE2_RESUME="${STAGE2_RESUME:-0}"

STAGE1_PARAMS="${CHECKPOINT_BASE_DIR}/pi05_libero_skill_router_stage1/${STAGE1_EXP}/${STAGE1_STEP}/params"
if [[ ! -d "${STAGE1_PARAMS}" ]]; then
  echo "Stage 1 params not found: ${STAGE1_PARAMS}" >&2
  exit 1
fi
export OPENPI_SKILL_STAGE1_PARAMS="${STAGE1_PARAMS}"

############################################
# Training (stage 2)
############################################
TRAIN_ARGS=(
  "${STAGE2_CONFIG}"
  --exp-name="${STAGE2_EXP}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --num-train-steps="${STAGE2_NUM_TRAIN_STEPS}"
  --keep-period None
)
[[ "${STAGE2_RESUME}" == "1" ]] && TRAIN_ARGS+=(--resume)

echo "[$(date -Iseconds)] === Variant=${VARIANT} config=${STAGE2_CONFIG} exp=${STAGE2_EXP} (resume=${STAGE2_RESUME}) ==="
python3 scripts/train.py "${TRAIN_ARGS[@]}"
echo "[$(date -Iseconds)] === Training done ==="

############################################
# Locate latest checkpoint for eval
############################################
STAGE2_DIR="${CHECKPOINT_BASE_DIR}/${STAGE2_CONFIG}/${STAGE2_EXP}"
LATEST_STEP=$(ls "${STAGE2_DIR}" | grep -E '^[0-9]+$' | sort -n | tail -1)
CKPT="${STAGE2_DIR}/${LATEST_STEP}"
echo "[$(date -Iseconds)] === Eval checkpoint: ${CKPT} ==="

############################################
# Policy server
############################################
PORT="${PORT:-8010}"
EXP_TAG="${EXP_TAG:-${EXP_TAG_DEFAULT}}"
RESULTS_LIB10="${RESULTS_LIB10:-results/${EXP_TAG}_libero10}"
RESULTS_LIBPRO="${RESULTS_LIBPRO:-results/${EXP_TAG}_libero_pro}"
mkdir -p "${RESULTS_LIB10}" "${RESULTS_LIBPRO}" "data/${EXP_TAG}_libero10/videos" "data/${EXP_TAG}_libero_pro/videos"

python3 -u scripts/serve_policy.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config="${STAGE2_CONFIG}" \
  --policy.dir="${CKPT}" \
  > "logs/${EXP_TAG}_policy_server.log" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} >/dev/null 2>&1 || true" EXIT
echo "[$(date -Iseconds)] === Policy server PID=${SERVER_PID} on port ${PORT}; waiting ${SERVER_READY_WAIT:-180}s ==="
sleep "${SERVER_READY_WAIT:-180}"

# Eval client lives in the gr00t env
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

SEEDS="${SEEDS:-7 42 100}"
REPLAN_STEPS="${REPLAN_STEPS:-10 5}"
NUM_TRIALS="${NUM_TRIALS:-10}"
PERTURB_TYPE="${PERTURB_TYPE:-object}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

result_path() {
  local dir="$1" seed="$2" h="$3"
  printf '%s/libero_eval_modelopenpi_tasklibero_10_seed%s_h%s.json' "${dir}" "${seed}" "${h}"
}

############################################
# libero_10
############################################
for seed in ${SEEDS}; do
  for h in ${REPLAN_STEPS}; do
    out=$(result_path "${RESULTS_LIB10}" "${seed}" "${h}")
    if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}" ]]; then
      echo "[$(date -Iseconds)] skip libero10 seed=${seed} h=${h}"
      continue
    fi
    echo "[$(date -Iseconds)] === libero10 seed=${seed} h=${h} ==="
    python examples/libero/main.py \
      --args.port "${PORT}" \
      --args.task-suite-name libero_10 \
      --args.seed "${seed}" \
      --args.replan-steps "${h}" \
      --args.action-horizon "${h}" \
      --args.num-trials-per-task "${NUM_TRIALS}" \
      --args.results-dir "${RESULTS_LIB10}" \
      --args.video-out-path "data/${EXP_TAG}_libero10/videos/seed${seed}_h${h}" \
      2>&1 | tee "logs/${EXP_TAG}_libero10_seed${seed}_h${h}.log"
  done
done

############################################
# libero_pro
############################################
for seed in ${SEEDS}; do
  for h in ${REPLAN_STEPS}; do
    out=$(result_path "${RESULTS_LIBPRO}" "${seed}" "${h}")
    if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}" ]]; then
      echo "[$(date -Iseconds)] skip libero_pro seed=${seed} h=${h}"
      continue
    fi
    echo "[$(date -Iseconds)] === libero_pro seed=${seed} h=${h} ==="
    python examples/libero/main_pro.py \
      --task_suite_name libero_10 \
      --perturbation_type "${PERTURB_TYPE}" \
      --seed "${seed}" \
      --replan_steps "${h}" \
      --action_horizon "${h}" \
      --num_trials_per_task "${NUM_TRIALS}" \
      --results_dir "${RESULTS_LIBPRO}" \
      --port "${PORT}" \
      --video_out_path "data/${EXP_TAG}_libero_pro/videos/seed${seed}_h${h}" \
      2>&1 | tee "logs/${EXP_TAG}_libero_pro_seed${seed}_h${h}.log"
  done
done

echo "[$(date -Iseconds)] === All done (variant=${VARIANT}) ==="
