# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager:** `uv` (required — do not use pip/conda directly)

**Setup:**
```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

**Run tests:**
```bash
uv run pytest --strict-markers -m "not manual"
# Single test file:
uv run pytest src/openpi/models/model_test.py --strict-markers -m "not manual"
```

**Lint and format:**
```bash
ruff check .
ruff format .
```

**Training:**
```bash
# JAX training (primary)
uv run python scripts/train.py <config_name> --exp-name <name>

# PyTorch training (multi-GPU)
torchrun --nproc_per_node=<N> scripts/train_pytorch.py --config <config_name>

# Serve a policy via WebSocket
uv run python scripts/serve_policy.py --env <env_name> --checkpoint-path <path>
```

**Normalization stats (required before training on new data):**
```bash
uv run python scripts/compute_norm_stats.py --config <config_name>
```

## Architecture

OpenPI is a Vision-Language-Action (VLA) robotics foundation model framework supporting three model variants: **π₀** (flow-matching), **π₀-FAST** (autoregressive), and **π₀.₅** (knowledge-insulated). The primary backend is JAX/Flax; PyTorch support is a newer addition.

### Dual Backend Design

- **JAX backend** (`src/openpi/models/`): Primary implementation using Flax NNX. Handles π₀, π₀-FAST, π₀.₅ with Optax optimizers and Orbax checkpointing.
- **PyTorch backend** (`src/openpi/models_pytorch/`): Newer parallel implementation with DDP/FSDP. Includes patches to HuggingFace Transformers in `transformers_replace/` for precision and KV-cache control.

### Model Architecture

All models share a common backbone:
- **Vision encoder:** SigLIP (`models/siglip.py`, `models/vit.py`)
- **Language backbone:** Google Gemma (`models/gemma.py` / `models/gemma_fast.py`)
- **Action head:** Flow-matching diffusion (π₀/π₀.₅) or autoregressive FAST tokenizer (π₀-FAST)
- **LoRA:** Low-rank adaptation layers in `models/lora.py` / `models_pytorch/lora_pytorch.py` for efficient finetuning

### Policy Layer

`src/openpi/policies/` contains robot-specific I/O adapters that sit above the model. Each policy (`aloha_policy.py`, `droid_policy.py`, `libero_policy.py`, etc.) converts raw robot observations/actions to/from the model's expected tensor format. The base `Policy` class in `policy.py` composes a model + a transform pipeline.

### Transform Pipeline

`src/openpi/transforms.py` implements a composable data transformation pipeline:
1. Robot-specific repacking (observation keys → model input keys)
2. Data-specific transforms (normalization, image resizing)
3. Model-specific transforms (tokenization, prompt injection, action chunking)

Transforms flow through `policy_config.py` which acts as the factory for assembling policies from checkpoints.

### Training Config System

`src/openpi/training/config.py` (~1300+ lines) is the single source of truth for 50+ pre-defined training configurations covering all model/robot combinations. Configs specify dataset, normalization, LoRA rank, optimizer, weight loading strategy, and more. Training entry points use **Tyro CLI** for runtime overrides.

### Serving Architecture

Remote inference uses a WebSocket server (`serving/websocket_policy_server.py`) and a lightweight pure-Python client library in `packages/openpi-client/`. The client has minimal dependencies (numpy, websockets, dm-tree) and exposes `WebsocketClientPolicy` for connecting to a remote policy server.

### Checkpoint Management

`src/openpi/shared/download.py` auto-downloads checkpoints from Google Cloud Storage with local caching. `training/weight_loaders.py` handles flexible weight initialization from base foundation model checkpoints when starting finetuning.

### Key Data Flows

- **Training data:** LeRobot datasets (via `training/data_loader.py`) and RLDS format (via `training/droid_rlds_dataset.py`)
- **Normalization:** Z-score or quantile normalization stats precomputed by `scripts/compute_norm_stats.py`, stored alongside checkpoints
- **Sharding:** FSDP setup in `training/sharding.py` for multi-device JAX training

## Notable Conventions

- Array type annotations use `src/openpi/shared/array_typing.py` — shapes are encoded in type hints
- π₀.₅ uses a "knowledge insulation" technique that separates language-following from motor skill training; see `docs/pi05_skill_router_plan.md` for experimental skill routing work
- The JAX model uses Flax NNX (not the older Linen API); `shared/nnx_utils.py` has utilities for NNX patterns
- JAX-to-PyTorch model conversion: `examples/convert_jax_model_to_pytorch.py`
