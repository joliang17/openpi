#!/bin/bash
#SBATCH --job-name=pi05_skill_router_ablation
#SBATCH --output=slurm_output/pi05_skill_router_ablation_%j.log
#SBATCH --error=slurm_output/pi05_skill_router_ablation_%j.log
#SBATCH --time=72:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G

# Stage-2 ablation training (no eval). Driven by VARIANT={adarms_only|film_only|film_vlm}.
# Submit:
#   sbatch --export=ALL,VARIANT=adarms_only shell_scripts/sbatch_pi05_skill_router_ablation.sh
#   sbatch --export=ALL,VARIANT=film_only   shell_scripts/sbatch_pi05_skill_router_ablation.sh
#   sbatch --export=ALL,VARIANT=film_vlm    shell_scripts/sbatch_pi05_skill_router_ablation.sh

set -euo pipefail

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai

cd /fs/nexus-scratch/yliang17/Research/VLA/openpi
mkdir -p slurm_output

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
case "${VARIANT:?set VARIANT=adarms_only|film_only|film_vlm}" in
  adarms_only)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_adarms_only
    STAGE2_EXP_DEFAULT=pi05_skill_router_stage2_adarms_only_${RUN_TAG}
    ;;
  film_only)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_film_only
    STAGE2_EXP_DEFAULT=pi05_skill_router_stage2_film_only_${RUN_TAG}
    ;;
  film_vlm)
    STAGE2_CONFIG=pi05_libero_skill_router_stage2_film_vlm
    STAGE2_EXP_DEFAULT=pi05_skill_router_stage2_film_vlm_${RUN_TAG}
    ;;
  *)
    echo "unknown VARIANT=${VARIANT}" >&2
    exit 1
    ;;
esac

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/fs/nexus-projects/wilddiffusion/vla/openpi_skill_router}"
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

TRAIN_ARGS=(
  "${STAGE2_CONFIG}"
  --exp-name="${STAGE2_EXP}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --num-train-steps="${STAGE2_NUM_TRAIN_STEPS}"
  --keep-period None
)
[[ "${STAGE2_RESUME}" == "1" ]] && TRAIN_ARGS+=(--resume)

echo "[$(date -Iseconds)] JOB=${SLURM_JOB_ID:-local} VARIANT=${VARIANT} CONFIG=${STAGE2_CONFIG} EXP=${STAGE2_EXP}"
echo "[$(date -Iseconds)] STAGE1_PARAMS=${STAGE1_PARAMS}"
python3 scripts/train.py "${TRAIN_ARGS[@]}"
echo "[$(date -Iseconds)] === Training done (variant=${VARIANT}) ==="
