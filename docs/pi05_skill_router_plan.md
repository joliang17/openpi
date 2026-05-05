# pi0.5 Skill Router + Skill Embedding Plan

## Summary

This experiment adds GR00T-style two-stage skill routing to OpenPI `pi05_base`, while using AtomicVLA-style LIBERO skill annotations for data loading.

The implementation keeps existing OpenPI configs unchanged. Skill routing is enabled only by `Pi0Config(use_skill_router=True, pi05=True, ...)`.

## Structure

- Data path:
  - `src/openpi/training/data_loader.py` adds `SkillAnnotatedDataset`.
  - The wrapper reads AtomicVLA/GR00T segment JSON and attaches `skill_id` and `skill_mask` to each LeRobot frame.
  - Default skill labels use `primary_action_verb` with vocabulary `["close", "open", "pick", "place", "turn"]`.

- Observation path:
  - `src/openpi/models/model.py` extends `Observation` with optional `skill_id` and `skill_mask`.
  - Normal pi0/pi0.5 observations keep these fields as `None`.

- Model path:
  - `src/openpi/models/pi0_config.py` adds gated skill-router config fields.
  - `src/openpi/models/pi0.py` adds:
    - `skill_pool_proj` and `skill_classifier` for stage-1 classification.
    - `skill_emb_bank` for learned skill embeddings.
    - `skill_to_adarms` for pi0.5 adaRMS conditioning.
    - `skill_to_action_film` for lightweight action-token modulation.

- Training path:
  - `scripts/train.py` now supports `compute_loss` returning `(loss, aux_info)` so skill metrics can be logged.
  - `src/openpi/training/weight_loaders.py` adds `PartialCheckpointWeightLoader` so new skill parameters can be initialized while loading `pi05_base`.

## Conditioning Design

GR00T injects skill information as an extra token into a DiT-style action module. pi0.5 is different: its action expert is a Gemma-style transformer that already receives flow timestep information through adaRMS.

For pi0.5, the better insertion point is:

```text
prefix tokens -> frozen VLM/action-prefix features -> skill classifier
skill probabilities -> weighted skill embedding
skill embedding -> adaRMS condition added to time condition
skill embedding -> optional FiLM on action tokens
```

This keeps action suffix length, attention masks, positional indexing, and KV-cache behavior unchanged. It also conditions every action-expert layer through pi0.5's native adaRMS path instead of adding a token that the action expert was not pretrained to interpret.

## Train Configs

- `pi05_libero_skill_router_stage1`
  - Loads `pi05_base`.
  - Trains only `skill_pool_proj` and `skill_classifier`.
  - Uses skill cross entropy only.

- `pi05_libero_skill_router_stage2`
  - Loads stage-1 params through `OPENPI_SKILL_STAGE1_PARAMS`.
  - Freezes VLM and classifier.
  - Trains only skill embedding/conditioning modules and the pi0.5 action expert:
    - `skill_emb_bank`
    - `skill_to_adarms`
    - `skill_to_action_film`
    - `llm.*_1`
    - `action_in_proj`
    - `action_out_proj`
    - `time_mlp_in`
    - `time_mlp_out`
  - Uses flow-matching action loss plus small skill embedding diversity/norm regularizers.

## Running

Use:

```bash
sbatch shell_scripts/run_pi05_libero_skill_router.sh
```

The skill annotation JSON is configured with `OPENPI_LIBERO_SKILL_ANNOTATION_PATH`. The provided shell script defaults it to the current cluster path; on another machine, set it to your local AtomicVLA/GR00T annotation JSON or place the file at `data_split_json/libero_lerobot_addskill_10_half.json`.

Before the first full run, compute normalization stats if they do not already exist for `libero_atomic`:

```bash
python3 scripts/compute_norm_stats.py --config-name pi05_libero_skill_router_stage1
```

The stage-2 command expects:

```bash
export OPENPI_SKILL_STAGE1_PARAMS=/path/to/stage1/checkpoint/params
```

The provided shell script sets this to the stage-1 output checkpoint path.

## Validation

- Run a one-batch data loader smoke test and confirm `skill_id` and `skill_mask` are present.
- Confirm stage-1 trainable parameters include only `skill_pool_proj` and `skill_classifier`.
- Confirm stage-2 trainable parameters include only skill-conditioning modules and action expert modules.
- Check logs:
  - Stage 1: `skill_cls_loss`, `skill_acc`.
  - Stage 2: `action_loss`, `skill_emb_div_loss`, `skill_emb_norm_loss`.
