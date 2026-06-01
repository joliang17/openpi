# LIBERO Skill Router Experiments

Last updated: 2026-06-01 00:03 America/New_York

## Current Evaluation Workflow

All evaluations use `shell_scripts/eval_pi05_skill_router_joint.sh`, with these defaults unless overridden:

- Suites: `libero10 libero_pro`
- LIBERO-PRO perturbation: `object`
- Seeds: `7 42 100`
- Replan/action horizons: `5 10`
- Trials per task: `10`
- Result folders: `results/${EXP_NAME}_libero10` and `results/${EXP_NAME}_libero_pro`
- Status folders: `data/${EXP_NAME}_libero10_eval_status` and `data/${EXP_NAME}_libero_pro_eval_status`
- Videos: `data/${EXP_NAME}_libero10/videos` and `data/${EXP_NAME}_libero_pro/videos`

The evaluation script starts a policy server in the `openai` conda environment, waits for warmup, switches to the `gr00t` conda environment, then runs `examples/libero/main.py` and `examples/libero/main_pro.py`.

## Saving and Bookkeeping Process

Use one stable experiment name per trained checkpoint, and reuse it consistently across checkpoint, result, status, video, and log paths.

Training output convention:

```text
/fs/nexus-projects/wilddiffusion/vla/<project_bucket>/<config_name>/<run_name>/<step>
```

For these runs the evaluated checkpoint step is `29999`, and each checkpoint folder contains:

```text
29999/_CHECKPOINT_METADATA
29999/assets/
29999/params/
29999/train_state/
wandb_id.txt
```

Evaluation output convention:

```text
results/${EXP_NAME}_libero10/
results/${EXP_NAME}_libero_pro/
data/${EXP_NAME}_libero10_eval_status/
data/${EXP_NAME}_libero_pro_eval_status/
data/${EXP_NAME}_libero10/videos/
data/${EXP_NAME}_libero_pro/videos/
logs/${EXP_NAME}_policy_server.log
logs/${EXP_NAME}_libero10_seed${seed}_steps${horizon}.log
logs/${EXP_NAME}_libero_pro_${PERTURBATION_TYPE}_seed${seed}_steps${horizon}.log
slurm_output/<job_name>_<job_id>.log
```

For future runs:

- Record checkpoint path, config name, `EXP_NAME`, job ID, port, and eval script in this file.
- Do not reuse ports across simultaneous eval jobs.
- Use `SKIP_COMPLETED_RESULTS=0` only when intentionally regenerating result JSONs/videos.
- If adding a new model folder name, update `MODEL_ALIASES` in `scripts/update_eval_summary_csv.py` before the CSV update job runs.
- After all eval jobs finish, run or schedule `python scripts/update_eval_summary_csv.py`.
- Keep generated result folders and `results_csv/` out of source commits unless explicitly asked to commit result artifacts.

After all current eval jobs finish successfully, CSV update is scheduled as:

```bash
python scripts/update_eval_summary_csv.py
```

The active dependent CSV job is:

```text
6950594 update_eval_csv afterok:6949045:6949046:6950593
```

## Evaluation Jobs Submitted

### VLM LoRA Skill-Effect Gate Router

```text
Job ID: 6949046
Job name: eval_effect_gate_router
EXP_NAME: pi05_vlm_lora_skill_effect_gate_router_joint
POLICY_CONFIG: pi05_libero_vlm_lora_skill_effect_gate_router_joint
PORT: 8031
Checkpoint:
/fs/nexus-projects/wilddiffusion/vla/openpi_vlm_lora_skill_effect_gate_router/pi05_libero_vlm_lora_skill_effect_gate_router_joint/pi05_vlm_lora_skill_effect_gate_router_joint_20260530_154123/29999
```

Early sanity check: server loaded and warmed up successfully; LIBERO10 seed 7 horizon 5 produced nonzero success rate.

### VLM LoRA Gated-FiLM Skill Router

```text
Job ID: 6949045
Job name: eval_gated_film_router
EXP_NAME: pi05_vlm_lora_gated_film_skill_router_joint
POLICY_CONFIG: pi05_libero_vlm_lora_gated_film_skill_router_joint
PORT: 8032
Checkpoint:
/fs/nexus-projects/wilddiffusion/vla/openpi_vlm_lora_gated_film_skill_router/pi05_libero_vlm_lora_gated_film_skill_router_joint/pi05_vlm_lora_gated_film_skill_router_joint_20260519_224324/29999
```

Early sanity check: server loaded and warmed up successfully; LIBERO10 seed 7 horizon 5 produced nonzero success rate.

### LoRA Gated-FiLM Skill Router

```text
Job ID: 6950593
Job name: eval_lora_gated_film_router
EXP_NAME: pi05_lora_gated_film_skill_router_joint
POLICY_CONFIG: pi05_libero_lora_gated_film_skill_router_joint
PORT: 8033
Checkpoint:
/fs/nexus-projects/wilddiffusion/vla/openpi_lora_gated_film_skill_router/pi05_libero_lora_gated_film_skill_router_joint/pi05_lora_gated_film_skill_router_joint_20260519_224324/29999
```

Submitted after the first two jobs. Check `logs/pi05_lora_gated_film_skill_router_joint_policy_server.log` and `slurm_output/eval_lora_gated_film_router_6950593.log` for startup/progress.

## Code Changes Used for These Evaluations

Committed as:

```text
0ff7762 Add skill gate overlays for LIBERO evals
```

Main changes:

- `Pi0.sample_actions_with_info(...)` returns actions plus skill diagnostics.
- `Policy.infer(...)` includes skill metadata in websocket inference responses.
- LIBERO and LIBERO-PRO videos overlay skill name, top-1 skill probability, and gate probability when available.
- `shell_scripts/eval_pi05_skill_router_joint.sh` accepts env-overridden `EXP_NAME` and `POLICY_CONFIG`.
- Slurm logs now use `%x_%j` to avoid collisions.
- `scripts/update_eval_summary_csv.py` aliases the new model folders for readable CSV rows.

Overlay fields:

- All skill-router models: `skill=<name> p=<top1_prob>`
- Gated-FiLM models: `action_gate=<prob>`
- Skill-effect-gate models: `effect_gate=<prob>`

Example verified frame overlays:

```text
skill=pick p=1.000
action_gate=0.034
```

and:

```text
skill=pick p=1.000
effect_gate=0.034
```

## Model Feature Differences

### `pi05_skill_router_joint`

Baseline VLM LoRA skill router.

- Config: `pi05_libero_vlm_lora_skill_router_joint`
- Uses `pi05=True`
- VLM uses `gemma_2b_lora`
- Action expert uses frozen base `gemma_300m`
- Uses skill classifier with 5 skills: `close`, `open`, `pick`, `place`, `turn`
- Jointly trains action prediction and skill classification
- Injects the routed soft skill embedding through the default skill-conditioning path
- No explicit action-FiLM gate
- No skill-effect gate
- Existing LIBERO10 results are in `results/pi05_skill_router_joint_libero10`

### `pi05_vlm_lora_skill_effect_gate_router_joint`

Adds a learned scalar gate controlling whether the routed skill embedding affects action generation.

- Config: `pi05_libero_vlm_lora_skill_effect_gate_router_joint`
- Uses `pi05=True`
- VLM uses `gemma_2b_lora`
- Action expert uses frozen base `gemma_300m`
- Uses skill classifier with the same 5-skill vocabulary
- Joint skill/action training
- `use_skill_effect_gate=True`
- `skill_effect_gate_source="skill_emb"`
- `skill_effect_gate_logit_bias=-2.0`
- Gate is applied as `skill_emb = effect_gate * skill_emb`
- Video overlay should show `effect_gate=<prob>`

Intent: learn when skill conditioning should be active or suppressed.

### `pi05_vlm_lora_gated_film_skill_router_joint`

Routes skill information through action-FiLM, with a learned scalar gate on that FiLM effect.

- Config: `pi05_libero_vlm_lora_gated_film_skill_router_joint`
- Uses `pi05=True`
- VLM uses `gemma_2b_lora`
- Action expert uses frozen base `gemma_300m`
- Uses skill classifier with the same 5-skill vocabulary
- Joint skill/action training
- `skill_inject_adarms=False`
- `use_skill_action_film=True`
- `use_skill_action_film_gate=True`
- `skill_inject_vlm_hidden=False`
- `skill_inject_state_token=False`
- Video overlay should show `action_gate=<prob>`

Intent: condition action-token processing with skill-dependent FiLM while learning when the FiLM signal should be used.

### `pi05_lora_gated_film_skill_router_joint`

Same gated-FiLM skill-router idea as above, but both the VLM and action expert use LoRA variants.

- Config: `pi05_libero_lora_gated_film_skill_router_joint`
- Uses `pi05=True`
- VLM uses `gemma_2b_lora`
- Action expert uses `gemma_300m_lora`
- Uses skill classifier with the same 5-skill vocabulary
- Joint skill/action training
- `skill_inject_adarms=False`
- `use_skill_action_film=True`
- `use_skill_action_film_gate=True`
- `skill_inject_vlm_hidden=False`
- `skill_inject_state_token=False`
- Video overlay should show `action_gate=<prob>`

Intent: test whether adding LoRA adaptation to the action expert improves the gated-FiLM skill-router variant.

## Follow-Up Analysis After Evaluation Finishes

When result JSONs exist for all runs, compare task-level LIBERO10 performance:

1. `pi05_vlm_lora_skill_effect_gate_router_joint_libero10` vs `pi05_vlm_lora_gated_film_skill_router_joint_libero10`
2. `pi05_vlm_lora_gated_film_skill_router_joint_libero10` vs `pi05_skill_router_joint_libero10`
3. Optionally compare `pi05_lora_gated_film_skill_router_joint_libero10` vs `pi05_vlm_lora_gated_film_skill_router_joint_libero10`

Suggested comparison procedure:

- For each result directory, read all `libero_eval_*.json`.
- Group by `action_horizon`.
- For each LIBERO10 task, average success rate across seeds.
- Report tasks where model A is better, worse, or tied against model B.
- Also report overall average deltas for horizon 5 and horizon 10.

Useful paths:

```text
results/pi05_vlm_lora_skill_effect_gate_router_joint_libero10
results/pi05_vlm_lora_gated_film_skill_router_joint_libero10
results/pi05_lora_gated_film_skill_router_joint_libero10
results/pi05_skill_router_joint_libero10
```
