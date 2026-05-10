import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.use_skill_router = config.use_skill_router
        self.skill_stage = config.skill_stage
        self.num_skills = config.num_skills
        self.use_skill_action_film = config.use_skill_action_film
        self.skill_inject_adarms = config.skill_inject_adarms
        self.skill_inject_vlm_hidden = config.skill_inject_vlm_hidden
        self.skill_inject_state_token = config.skill_inject_state_token
        self.skill_clf_loss_weight = config.skill_clf_loss_weight
        self.skill_emb_div_loss_weight = config.skill_emb_div_loss_weight
        self.skill_emb_norm_loss_weight = config.skill_emb_norm_loss_weight
        self.knowledge_insulation = config.knowledge_insulation
        self.ki_loss_weight = config.ki_loss_weight
        self.action_loss_weight = config.action_loss_weight
        if config.use_skill_router:
            self.skill_pool_proj = nnx.Linear(
                paligemma_config.width, config.skill_router_hidden_dim, rngs=rngs
            )
            self.skill_classifier = nnx.Linear(config.skill_router_hidden_dim, config.num_skills, rngs=rngs)
            self.skill_emb_bank = nnx.Param(
                jax.random.normal(rngs.params(), (config.num_skills, config.skill_emb_dim)) * 0.02
            )
            if config.skill_inject_adarms:
                self.skill_to_adarms = nnx.Linear(config.skill_emb_dim, action_expert_config.width, rngs=rngs)
            if config.use_skill_action_film:
                self.skill_to_action_film = nnx.Linear(config.skill_emb_dim, 2 * action_expert_config.width, rngs=rngs)
            if config.skill_inject_vlm_hidden:
                self.skill_to_vlm_hidden = nnx.Linear(config.skill_emb_dim, paligemma_config.width, rngs=rngs)
            if config.skill_inject_state_token:
                self.skill_to_state = nnx.Linear(config.skill_emb_dim, action_expert_config.width, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _classify_skill(
        self, prefix_out: at.Float[at.Array, "b s emb"], prefix_mask: at.Bool[at.Array, "b s"]
    ) -> at.Float[at.Array, "b n"]:
        mask = prefix_mask.astype(prefix_out.dtype)[..., None]
        pooled = jnp.sum(prefix_out * mask, axis=1) / jnp.clip(jnp.sum(mask, axis=1), 1.0)
        hidden = self.skill_pool_proj(pooled)
        hidden = nnx.swish(hidden)
        return self.skill_classifier(hidden)

    def _skill_condition(
        self, skill_logits: at.Float[at.Array, "b n"]
    ) -> tuple[
        at.Float[at.Array, "b emb"] | None,
        at.Float[at.Array, "b emb2"] | None,
        at.Float[at.Array, "b emb3"] | None,
        at.Float[at.Array, "b emb4"] | None,
    ]:
        skill_probs = jax.nn.softmax(skill_logits, axis=-1)
        skill_emb = skill_probs @ self.skill_emb_bank.value
        adarms_skill_cond = self.skill_to_adarms(skill_emb) if self.skill_inject_adarms else None
        action_film = self.skill_to_action_film(skill_emb) if self.use_skill_action_film else None
        vlm_delta = self.skill_to_vlm_hidden(skill_emb) if self.skill_inject_vlm_hidden else None
        state_cond = self.skill_to_state(skill_emb) if self.skill_inject_state_token else None
        return adarms_skill_cond, action_film, vlm_delta, state_cond

    def _skill_losses(
        self, skill_logits: at.Float[at.Array, "b n"], obs: _model.Observation
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        if obs.skill_id is None:
            zeros = jnp.zeros(skill_logits.shape[0], dtype=skill_logits.dtype)
            return zeros, {
                "skill_cls_loss": jnp.asarray(0.0, dtype=skill_logits.dtype),
                "skill_acc": jnp.asarray(0.0, dtype=skill_logits.dtype),
            }
        skill_mask = (
            jnp.ones(skill_logits.shape[0], dtype=jnp.bool_)
            if obs.skill_mask is None
            else obs.skill_mask.astype(jnp.bool_)
        )
        labels = jnp.clip(obs.skill_id.astype(jnp.int32), 0, self.num_skills - 1)
        per_example = -jnp.sum(jax.nn.one_hot(labels, self.num_skills) * jax.nn.log_softmax(skill_logits), axis=-1)
        per_example = jnp.where(skill_mask, per_example, 0.0)
        denom = jnp.clip(jnp.sum(skill_mask), 1)
        mean_ce = jnp.sum(per_example) / denom
        correct = (jnp.argmax(skill_logits, axis=-1) == labels).astype(skill_logits.dtype)
        acc = jnp.sum(jnp.where(skill_mask, correct, 0.0)) / denom
        return per_example, {"skill_cls_loss": mean_ce, "skill_acc": acc}

    def _skill_embedding_regularizers(self) -> dict[str, at.Array]:
        emb = self.skill_emb_bank.value
        emb_norm = emb / jnp.clip(jnp.linalg.norm(emb, axis=-1, keepdims=True), 1e-6)
        gram = emb_norm @ emb_norm.T
        off_diag = gram - jnp.eye(self.num_skills, dtype=gram.dtype)
        div_loss = jnp.mean(jnp.square(off_diag))
        norm_loss = jnp.mean(1.0 / (jnp.square(jnp.linalg.norm(emb, axis=-1)) + 1e-6))
        return {"skill_emb_div_loss": div_loss, "skill_emb_norm_loss": norm_loss}

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_ki_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask.append(jnp.zeros(input_mask[-1].shape, dtype=jnp.int32))

        assert obs.ki_tokenized_prompt is not None, "KI tokenized prompt is required"
        assert obs.ki_tokenized_prompt_mask is not None, "KI tokenized prompt mask is required"
        assert obs.ki_token_ar_mask is not None, "KI token auto-regressive mask is required"
        tokenized_inputs = self.PaliGemma.llm(obs.ki_tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.ki_tokenized_prompt_mask)
        ar_mask.append(obs.ki_token_ar_mask)

        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.concatenate(ar_mask, axis=1)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        skill_adarms_cond: at.Float[at.Array, "b emb"] | None = None,
        skill_action_film: at.Float[at.Array, "b emb2"] | None = None,
        skill_state_cond: at.Float[at.Array, "b emb3"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            if skill_state_cond is not None:
                state_token = state_token + skill_state_cond[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if skill_action_film is not None:
            gamma, beta = jnp.split(skill_action_film, 2, axis=-1)
            action_tokens = action_tokens * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            if skill_adarms_cond is not None:
                time_emb = time_emb + skill_adarms_cond
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_ki_ce_loss(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        assert observation.ki_tokenized_prompt is not None, "KI tokenized prompt is required"
        assert observation.ki_token_loss_mask is not None, "KI token loss mask is required"

        input_token_embeddings, input_mask, ar_mask = self.embed_ki_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        targets = observation.ki_tokenized_prompt[:, 1:]
        target_one_hot = jax.nn.one_hot(targets, _gemma.PALIGEMMA_VOCAB_SIZE)
        (pre_logits, _), _ = self.PaliGemma.llm(
            [input_token_embeddings[:, :-1], None],
            mask=attn_mask[:, :-1, :-1],
            positions=positions[:, :-1],
            adarms_cond=[None, None],
        )
        logits = self.PaliGemma.llm(pre_logits[:, -targets.shape[1] :], method="decode")
        logp = jax.nn.log_softmax(logits, axis=-1)

        loss_mask = observation.ki_token_loss_mask[:, 1:]
        token_logp = jnp.sum(target_one_hot * logp, axis=-1)
        denom = jnp.clip(jnp.sum(loss_mask, axis=-1), 1)
        ce_loss = -jnp.sum(token_logp * loss_mask, axis=-1) / denom
        token_accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == targets) * loss_mask, axis=-1) / denom
        return ce_loss, {
            "ki_ce_loss": jnp.mean(ce_loss),
            "ki_token_accuracy": jnp.mean(token_accuracy),
        }

    def _compute_insulated_action_loss(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        time: at.Float[at.Array, " b"],
        u_t: _model.Actions,
    ) -> tuple[at.Float[at.Array, "b ah"], dict[str, at.Array]]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), prefix_kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None],
        )
        prefix_out = jax.lax.stop_gradient(prefix_out)
        prefix_kv_cache = jax.lax.stop_gradient(prefix_kv_cache)

        skill_logits = None
        skill_adarms_cond = None
        skill_action_film = None
        skill_state_cond = None
        if self.use_skill_router:
            skill_logits = self._classify_skill(prefix_out, prefix_mask)
            skill_adarms_cond, skill_action_film, vlm_delta, skill_state_cond = self._skill_condition(skill_logits)
            if self.skill_stage == "classifier":
                skill_loss, skill_info = self._skill_losses(skill_logits, observation)
                skill_loss = einops.repeat(skill_loss, "b -> b s", s=self.action_horizon)
                return self.skill_clf_loss_weight * skill_loss, skill_info
            if vlm_delta is not None:
                prefix_tokens_cond = prefix_tokens + vlm_delta[:, None, :]
                _, prefix_kv_cache = self.PaliGemma.llm(
                    [prefix_tokens_cond, None],
                    mask=prefix_attn_mask,
                    positions=prefix_positions,
                    adarms_cond=[None, None],
                )
                prefix_kv_cache = jax.lax.stop_gradient(prefix_kv_cache)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, skill_adarms_cond, skill_action_film, skill_state_cond
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (unused_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=prefix_kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert unused_prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        info = {"action_loss": jnp.mean(action_loss)}
        if self.use_skill_router:
            reg = self._skill_embedding_regularizers()
            _, skill_info = self._skill_losses(skill_logits, observation)
            action_loss = (
                action_loss
                + self.skill_emb_div_loss_weight * reg["skill_emb_div_loss"]
                + self.skill_emb_norm_loss_weight * reg["skill_emb_norm_loss"]
            )
            info = {**info, **skill_info, **reg}
        return action_loss, info

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"] | tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.knowledge_insulation:
            action_loss, action_info = self._compute_insulated_action_loss(observation, x_t, time, u_t)
            ki_ce_loss, ki_info = self._compute_ki_ce_loss(observation)
            total_loss = self.action_loss_weight * action_loss + self.ki_loss_weight * ki_ce_loss[:, None]
            return total_loss, {**action_info, **ki_info}

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        skill_logits = None
        skill_adarms_cond = None
        skill_action_film = None
        skill_state_cond = None
        vlm_delta = None
        if self.use_skill_router:
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out_for_skill, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions, adarms_cond=[None, None]
            )
            skill_logits = self._classify_skill(prefix_out_for_skill, prefix_mask)
            skill_adarms_cond, skill_action_film, vlm_delta, skill_state_cond = self._skill_condition(skill_logits)
            if self.skill_stage == "classifier":
                skill_loss, skill_info = self._skill_losses(skill_logits, observation)
                skill_loss = einops.repeat(skill_loss, "b -> b s", s=self.action_horizon)
                return self.skill_clf_loss_weight * skill_loss, skill_info

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, skill_adarms_cond, skill_action_film, skill_state_cond
        )
        joint_prefix_tokens = prefix_tokens if vlm_delta is None else prefix_tokens + vlm_delta[:, None, :]
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [joint_prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not self.use_skill_router:
            return action_loss

        reg = self._skill_embedding_regularizers()
        _, skill_info = self._skill_losses(skill_logits, observation)
        total_loss = (
            action_loss
            + self.skill_emb_div_loss_weight * reg["skill_emb_div_loss"]
            + self.skill_emb_norm_loss_weight * reg["skill_emb_norm_loss"]
        )
        info = {
            "action_loss": jnp.mean(action_loss),
            **skill_info,
            **reg,
        }
        return total_loss, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        skill_adarms_cond = None
        skill_action_film = None
        skill_state_cond = None
        if self.use_skill_router:
            skill_logits = self._classify_skill(prefix_out, prefix_mask)
            skill_adarms_cond, skill_action_film, vlm_delta, skill_state_cond = self._skill_condition(skill_logits)
            if vlm_delta is not None:
                prefix_tokens_cond = prefix_tokens + vlm_delta[:, None, :]
                _, kv_cache = self.PaliGemma.llm(
                    [prefix_tokens_cond, None], mask=prefix_attn_mask, positions=positions
                )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), skill_adarms_cond, skill_action_film, skill_state_cond
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
