import dataclasses
import os
import pathlib

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


def test_metric_logging_helpers_handle_non_numeric_values():
    metrics = {
        "loss": jnp.asarray([1.0, 3.0]),
        "skill_acc": np.asarray(0.25, dtype=np.float32),
        "bf16_skill_acc": jnp.asarray(0.30078125, dtype=jnp.bfloat16),
        "stage": "action",
    }

    stacked = common_utils.stack_forest([metrics, metrics])
    reduced = jax.device_get(jax.tree.map(train._reduce_metric, stacked))  # noqa: SLF001

    assert float(reduced["loss"]) == 2.0
    assert (  # noqa: SLF001
        train._format_metrics(reduced)
        == "bf16_skill_acc=0.30078125, loss=2.0000, skill_acc=0.2500, stage=action"
    )
    wandb_metrics = train._wandb_metrics(reduced)  # noqa: SLF001
    assert wandb_metrics.keys() == {"loss", "skill_acc", "bf16_skill_acc"}
    assert wandb_metrics["loss"] == 2.0
    assert wandb_metrics["skill_acc"] == np.float32(0.25)
    assert wandb_metrics["bf16_skill_acc"] == 0.30078125
    assert isinstance(wandb_metrics["skill_acc"], float)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
