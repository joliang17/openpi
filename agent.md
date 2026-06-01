# Codex Runbook for LIBERO Training and Evaluation

This file records the process future Codex sessions should follow for OpenPI LIBERO skill-router experiments in this repo.

## Ground Rules

- Work from `/fs/nexus-scratch/yliang17/Research/VLA/openpi`.
- Check `git status --short` before edits, commits, and submissions.
- Do not revert user changes or unrelated untracked files.
- Use `rg` / `rg --files` for repo search.
- Use `apply_patch` for manual file edits.
- Commit only the files relevant to the requested code/documentation change.
- Treat `results/`, `data/`, `logs/`, `slurm_output/`, and `results_csv/` as generated outputs unless the user explicitly asks to commit artifacts.

## TODO Maintenance

Maintain `todo.md` as the project task queue.

- When the user gives a new follow-up task that cannot be completed immediately, add it to `todo.md`.
- When a task in `todo.md` is completed, delete that item from `todo.md`.
- Keep entries concrete and checkable, with relevant job IDs, paths, or model names when useful.
- Before ending work on a multi-step request, check `todo.md` and update it to reflect what remains.
- Commit `todo.md` updates when they are part of a requested documentation/process change. Do not commit generated result artifacts with it unless explicitly asked.

## Training Process

Training scripts live in `shell_scripts/train_*.sh`.

Before launching training:

1. Confirm the training config exists in `src/openpi/training/config.py`.
2. Confirm the training script exports the intended `CONFIG_NAME`, `EXP_NAME`, and `CHECKPOINT_BASE_DIR`.
3. Confirm skill annotations when using skill-router configs:

```bash
OPENPI_LIBERO_SKILL_ANNOTATION_PATH=/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json
```

Training output should be saved under:

```text
/fs/nexus-projects/wilddiffusion/vla/<project_bucket>/<config_name>/<run_name>/
```

Expected checkpoint layout:

```text
<run_name>/29999/_CHECKPOINT_METADATA
<run_name>/29999/assets/
<run_name>/29999/params/
<run_name>/29999/train_state/
<run_name>/wandb_id.txt
```

After training:

- Record the run name, config, checkpoint path, and main model feature difference in `experiment.md`.
- Use the explicit checkpoint step path, usually `<run_name>/29999`, for evaluation.

## Evaluation Process

Preferred eval script:

```text
shell_scripts/eval_pi05_skill_router_joint.sh
```

It supports environment overrides for:

- `EXP_NAME`
- `POLICY_CONFIG`
- `CKPT`
- `PORT`
- `SUITES`
- `SEEDS`
- `REPLAN_STEPS`
- `PERTURBATION_TYPE`
- `NUM_TRIALS_PER_TASK`
- `SKIP_COMPLETED_RESULTS`

Default full eval settings:

```text
SUITES="libero10 libero_pro"
PERTURBATION_TYPE="object"
SEEDS="7 42 100"
REPLAN_STEPS="5 10"
NUM_TRIALS_PER_TASK=10
```

Submit with unique job name and port:

```bash
sbatch --job-name=<job_name> \
  --export=ALL,EXP_NAME=<exp_name>,POLICY_CONFIG=<policy_config>,CKPT=<checkpoint_step_path>,PORT=<free_port> \
  shell_scripts/eval_pi05_skill_router_joint.sh
```

Example:

```bash
sbatch --job-name=eval_lora_gated_film_router \
  --export=ALL,EXP_NAME=pi05_lora_gated_film_skill_router_joint,POLICY_CONFIG=pi05_libero_lora_gated_film_skill_router_joint,CKPT=/fs/nexus-projects/wilddiffusion/vla/openpi_lora_gated_film_skill_router/pi05_libero_lora_gated_film_skill_router_joint/pi05_lora_gated_film_skill_router_joint_20260519_224324/29999,PORT=8033 \
  shell_scripts/eval_pi05_skill_router_joint.sh
```

Monitor startup:

```bash
squeue -j <job_id> -o '%.18i %.30j %.10T %.10M %.20R'
tail -n 80 logs/${EXP_NAME}_policy_server.log
tail -n 80 slurm_output/<job_name>_<job_id>.log
```

A healthy job should show:

- checkpoint restore finished
- norm stats loaded
- warm-up complete
- websocket server listening
- LIBERO task logs with nonzero rollout progress

Monitor status files:

```bash
cat data/${EXP_NAME}_libero10_eval_status/libero10_seed7_steps5.json
cat data/${EXP_NAME}_libero_pro_eval_status/libero_pro_object_seed7_steps5.json
```

Monitor result counts:

```bash
find results/${EXP_NAME}_libero10 -maxdepth 1 -type f -name 'libero_eval_*.json' | wc -l
find results/${EXP_NAME}_libero_pro -maxdepth 1 -type f -name 'libero_eval_*.json' | wc -l
```

For the default matrix, expect six LIBERO10 JSONs and six LIBERO-PRO JSONs per model.

## Video Overlay Checks

Skill-router eval videos should contain overlays:

- All skill-router models: `skill=<name> p=<prob>`
- Gated-FiLM variants: `action_gate=<prob>`
- Skill-effect-gate variants: `effect_gate=<prob>`

If `ffmpeg` is unavailable, inspect a frame with OpenCV:

```bash
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t
python - <<'PY'
from pathlib import Path
import cv2

video = "data/<exp_name>_libero10/videos/seed7_steps5/<some_rollout>.mp4"
out = Path("/tmp/openpi_overlay_check.jpg")
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError(video)
cv2.imwrite(str(out), frame)
print(out)
PY
```

Then inspect the saved frame with the local image viewer.

## Updating Result CSV

CSV updater:

```bash
python scripts/update_eval_summary_csv.py
```

If evals are still running, schedule the CSV update after all eval jobs:

```bash
sbatch --job-name=update_eval_csv \
  --output=slurm_output/update_eval_csv_%j.log \
  --error=slurm_output/update_eval_csv_%j.log \
  --time=00:30:00 \
  --account=scavenger \
  --partition=scavenger \
  --dependency=afterok:<job_a>:<job_b>:<job_c> \
  --wrap='cd /fs/nexus-scratch/yliang17/Research/VLA/openpi && python scripts/update_eval_summary_csv.py'
```

If a new eval job is added after a dependent CSV job was already queued:

```bash
scancel <old_csv_job_id>
sbatch ... --dependency=afterok:<all_eval_jobs> ...
```

If adding a new model family, first update `MODEL_ALIASES` in `scripts/update_eval_summary_csv.py` so `results_csv/eval_summary.csv` uses readable names.

## Committing Code Changes

Before committing:

```bash
git status --short
git diff --check
python -m py_compile <changed_python_files>
bash -n <changed_shell_scripts>
```

If Python package imports require the project conda environment:

```bash
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate openai
PYTHONPATH=src python - <<'PY'
from openpi.training import config
print(config.get_config("<policy_config>").name)
PY
```

Commit only relevant files:

```bash
git add <file1> <file2> ...
git commit -m "<concise message>"
```

Do not include generated eval outputs unless the user explicitly asks.

## Post-Eval Comparison Process

For LIBERO10 task-level comparisons:

1. Load all `libero_eval_*.json` from each model result directory.
2. Group by `action_horizon`.
3. Average each task success rate across seeds.
4. Report per-task better/worse/tied deltas.
5. Report overall average by horizon.

Important result directories for current comparisons:

```text
results/pi05_vlm_lora_skill_effect_gate_router_joint_libero10
results/pi05_vlm_lora_gated_film_skill_router_joint_libero10
results/pi05_lora_gated_film_skill_router_joint_libero10
results/pi05_skill_router_joint_libero10
```
