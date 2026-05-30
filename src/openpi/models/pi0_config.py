import dataclasses
from typing import Any, TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # If true, train pi0.5 with knowledge insulation: the action expert receives stopped-gradient VLM
    # features while the VLM is adapted with an auxiliary FAST-token action prediction objective.
    knowledge_insulation: bool = False
    ki_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    fast_model_tokenizer: Any | None = None
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    # Optional pi0.5 skill-router experiment. Disabled by default.
    use_skill_router: bool = False
    num_skills: int = 0
    skill_emb_dim: int = 256
    skill_router_hidden_dim: int = 256
    skill_stage: str = "action"  # "classifier", "action", or "joint"
    skill_clf_loss_weight: float = 1.0
    skill_emb_div_loss_weight: float = 0.01
    skill_emb_norm_loss_weight: float = 0.001
    use_skill_effect_gate: bool = False
    skill_effect_gate_source: str = "skill_emb"  # "skill_emb" or "prefix_hidden"
    skill_effect_gate_logit_bias: float = 0.0
    skill_effect_gate_eval_mode: str = "normal"  # "normal", "zero", or "one"; inference only
    use_skill_action_film: bool = True
    use_skill_action_film_gate: bool = False
    skill_action_gate_logit_bias: float = 0.0
    skill_inject_adarms: bool = True       # add skill to time_emb for adaRMS (pi05 only)
    skill_inject_vlm_hidden: bool = False  # add skill to VLM prefix input embeddings
    skill_inject_state_token: bool = False # add skill to state token in suffix (pi0 only)
    # Inference-time skill ablation (only affects sample_actions, not training):
    #   "normal"  -> use the routed skill embedding;
    #   "shuffle" -> randomly permute the skill_emb_bank rows each call, so the
    #                router's selected skill maps to a different skill's embedding;
    #   "zero"    -> drop skill conditioning entirely (skill embedding = 0);
    #   "gate_zero" -> force the action-FiLM skill gate closed;
    #   "gate_one"  -> force the action-FiLM skill gate open.
    # Used to test whether the skill routing actually contributes to performance.
    skill_eval_mode: str = "normal"

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.knowledge_insulation and not self.pi05:
            raise ValueError("knowledge_insulation is implemented for pi0.5 only.")
        if self.use_skill_router:
            if self.num_skills <= 0:
                raise ValueError("num_skills must be positive when use_skill_router=True.")
            if self.skill_stage not in ("classifier", "action", "joint"):
                raise ValueError("skill_stage must be 'classifier', 'action', or 'joint'.")
            if self.skill_eval_mode not in ("normal", "shuffle", "zero", "gate_zero", "gate_one"):
                raise ValueError("skill_eval_mode must be 'normal', 'shuffle', 'zero', 'gate_zero', or 'gate_one'.")
            if self.skill_effect_gate_source not in ("skill_emb", "prefix_hidden"):
                raise ValueError("skill_effect_gate_source must be 'skill_emb' or 'prefix_hidden'.")
            if self.skill_effect_gate_eval_mode not in ("normal", "zero", "one"):
                raise ValueError("skill_effect_gate_eval_mode must be 'normal', 'zero', or 'one'.")
            if self.skill_effect_gate_eval_mode != "normal" and not self.use_skill_effect_gate:
                raise ValueError("skill_effect_gate_eval_mode ablations require use_skill_effect_gate=True.")
            if self.skill_inject_adarms and not self.pi05:
                raise ValueError("skill_inject_adarms requires pi05=True (adaRMS is only used in pi0.5).")
            if self.skill_inject_state_token and self.pi05:
                raise ValueError("skill_inject_state_token requires pi05=False (pi0.5 has no state token in the suffix).")
            if self.use_skill_action_film_gate and not self.use_skill_action_film:
                raise ValueError("use_skill_action_film_gate requires use_skill_action_film=True.")
            if self.skill_eval_mode in ("gate_zero", "gate_one") and not self.use_skill_action_film_gate:
                raise ValueError("skill_eval_mode gate ablations require use_skill_action_film_gate=True.")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                ki_tokenized_prompt=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.knowledge_insulation
                    else None
                ),
                ki_tokenized_prompt_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool)
                    if self.knowledge_insulation
                    else None
                ),
                ki_token_ar_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.knowledge_insulation
                    else None
                ),
                ki_token_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_)
                    if self.knowledge_insulation
                    else None
                ),
                skill_id=jax.ShapeDtypeStruct([batch_size], jnp.int32) if self.use_skill_router else None,
                skill_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.use_skill_router else None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
