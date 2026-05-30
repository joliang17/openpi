from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi05_ki_dummy_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        knowledge_insulation=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=2,
        action_horizon=2,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    obs = obs.replace(
        image_masks={k: jnp.ones((batch_size,), dtype=jnp.bool_) for k in obs.images},
        tokenized_prompt=jnp.ones((batch_size, config.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_),
        ki_tokenized_prompt=jnp.ones((batch_size, config.max_token_len), dtype=jnp.int32),
        ki_tokenized_prompt_mask=jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_),
        ki_token_ar_mask=jnp.concatenate(
            [
                jnp.zeros((batch_size, config.max_token_len // 2), dtype=jnp.int32),
                jnp.ones((batch_size, config.max_token_len // 2), dtype=jnp.int32),
            ],
            axis=1,
        ),
        ki_token_loss_mask=jnp.concatenate(
            [
                jnp.zeros((batch_size, config.max_token_len // 2), dtype=jnp.bool_),
                jnp.ones((batch_size, config.max_token_len // 2), dtype=jnp.bool_),
            ],
            axis=1,
        ),
    )

    loss, info = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert "action_loss" in info
    assert "ki_ce_loss" in info
    assert "ki_token_accuracy" in info


def test_pi05_gated_film_skill_router_dummy_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=2,
        action_horizon=2,
        max_token_len=16,
        use_skill_router=True,
        num_skills=5,
        skill_emb_dim=8,
        skill_stage="joint",
        use_skill_effect_gate=True,
        skill_effect_gate_logit_bias=-2.0,
        skill_inject_adarms=False,
        use_skill_action_film=True,
        use_skill_action_film_gate=True,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    obs = obs.replace(
        skill_id=jnp.asarray([0, 1], dtype=jnp.int32),
        skill_mask=jnp.ones((batch_size,), dtype=jnp.bool_),
    )

    loss, info = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert "skill_action_gate_mean" in info
    assert "skill_action_gate_std" in info
    assert "skill_action_gate_skill_0" in info
    assert "skill_effect_gate_mean" in info
    assert "skill_effect_gate_std" in info
    assert "skill_effect_gate_skill_0" in info


def test_pi05_skill_effect_gate_config_validation():
    with pytest.raises(ValueError, match="skill_effect_gate_source"):
        pi0_config.Pi0Config(
            pi05=True,
            use_skill_router=True,
            num_skills=5,
            use_skill_effect_gate=True,
            skill_effect_gate_source="bad",
        )

    with pytest.raises(ValueError, match="skill_effect_gate_eval_mode"):
        pi0_config.Pi0Config(
            pi05=True,
            use_skill_router=True,
            num_skills=5,
            skill_effect_gate_eval_mode="bad",
        )

    with pytest.raises(ValueError, match="use_skill_effect_gate"):
        pi0_config.Pi0Config(
            pi05=True,
            use_skill_router=True,
            num_skills=5,
            skill_effect_gate_eval_mode="zero",
        )


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
