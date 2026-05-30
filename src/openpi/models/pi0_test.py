import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config
import openpi.shared.nnx_utils as nnx_utils


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def _get_filtered_state(config: _pi0_config.Pi0Config, state_filter: nnx.filterlib.Filter) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    return nnx.state(abstract_model, nnx.All(nnx.Param, state_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi05_freeze_vlm_filter():
    config = _pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    freeze_filter = nnx.Any(
        nnx_utils.PathRegex(".*img.*"),
        nnx.All(
            nnx_utils.PathRegex(".*llm.*"),
            nnx.Not(nnx_utils.PathRegex(".*_1.*")),
        ),
    )

    frozen_state = _get_filtered_state(config, freeze_filter)
    assert len(frozen_state) > 0
    assert any("img" in path for path in frozen_state)
    assert any("llm" in path for path in frozen_state)
    assert all(
        "img" in path or ("llm" in path and not any("_1" in part for part in path)) for path in frozen_state
    )

    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    trainable_state = _get_filtered_state(config, trainable_filter)
    assert any(any("_1" in part for part in path) for path in trainable_state)
    assert any("action_in_proj" in path for path in trainable_state)
    assert any("time_mlp_in" in path for path in trainable_state)
    assert any("time_mlp_out" in path for path in trainable_state)
    assert any("action_out_proj" in path for path in trainable_state)


def test_pi05_ki_vlm_lora_action_expert_filter():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=True,
        knowledge_insulation=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )
    freeze_filter = nnx.Any(
        nnx_utils.PathRegex(".*img.*"),
        nnx.All(
            nnx_utils.PathRegex(".*llm.*"),
            nnx.Not(nnx_utils.PathRegex(".*_1.*")),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
    )

    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    trainable_state = _get_filtered_state(config, trainable_filter)
    frozen_state = _get_filtered_state(config, freeze_filter)

    assert any(any("lora" in part for part in path) for path in trainable_state)
    assert any(any("_1" in part for part in path) for path in trainable_state)
    assert any("action_in_proj" in path for path in trainable_state)
    assert any("action_out_proj" in path for path in trainable_state)
    assert any("img" in path for path in frozen_state)
    assert all(
        not (
            "llm" in path
            and not any("_1" in part for part in path)
            and not any("lora" in part for part in path)
        )
        for path in trainable_state
    )


def test_pi05_gated_film_vlm_lora_filter():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        use_skill_router=True,
        num_skills=5,
        skill_stage="joint",
        skill_inject_adarms=False,
        use_skill_action_film=True,
        use_skill_action_film_gate=True,
    )
    freeze_filter = nnx.Any(
        nnx_utils.PathRegex(".*img.*"),
        nnx.All(
            nnx_utils.PathRegex(".*llm.*"),
            nnx.Not(nnx_utils.PathRegex(".*_1.*")),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
    )

    trainable_state = _get_filtered_state(config, nnx.All(nnx.Param, nnx.Not(freeze_filter)))
    frozen_state = _get_filtered_state(config, freeze_filter)

    assert any("skill_to_action_film_gate" in part for path in trainable_state for part in path)
    assert any("skill_to_action_film" in part for path in trainable_state for part in path)
    assert not any("skill_to_adarms" in part for path in trainable_state for part in path)
    assert any("img" in path for path in frozen_state)


def test_pi05_skill_effect_gate_vlm_lora_filter():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        use_skill_router=True,
        num_skills=5,
        skill_stage="joint",
        use_skill_effect_gate=True,
        skill_effect_gate_source="skill_emb",
    )
    freeze_filter = nnx.All(
        nnx.Param,
        nnx.Not(
            nnx.Any(
                nnx_utils.PathRegex(".*lora.*"),
                nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx_utils.PathRegex(".*action_in_proj.*"),
                nnx_utils.PathRegex(".*action_out_proj.*"),
                nnx_utils.PathRegex(".*time_mlp_in.*"),
                nnx_utils.PathRegex(".*time_mlp_out.*"),
                nnx_utils.PathRegex(".*skill_pool_proj.*"),
                nnx_utils.PathRegex(".*skill_classifier.*"),
                nnx_utils.PathRegex(".*skill_emb_bank.*"),
                nnx_utils.PathRegex(".*skill_effect_gate.*"),
                nnx_utils.PathRegex(".*skill_to_adarms.*"),
                nnx_utils.PathRegex(".*skill_to_action_film.*"),
            )
        ),
    )

    trainable_state = _get_filtered_state(config, nnx.All(nnx.Param, nnx.Not(freeze_filter)))

    assert any("skill_effect_gate" in part for path in trainable_state for part in path)
    assert any("skill_to_action_film" in part for path in trainable_state for part in path)
    assert any("skill_to_adarms" in part for path in trainable_state for part in path)


def test_pi05_gated_film_lora_filter_keeps_image_encoder_trainable():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        use_skill_router=True,
        num_skills=5,
        skill_stage="joint",
        skill_inject_adarms=False,
        use_skill_action_film=True,
        use_skill_action_film_gate=True,
    )

    trainable_state = _get_filtered_state(config, nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter())))

    assert any("skill_to_action_film_gate" in part for path in trainable_state for part in path)
    assert any("img" in path for path in trainable_state)
    assert any(any("lora" in part for part in path) for path in trainable_state)
