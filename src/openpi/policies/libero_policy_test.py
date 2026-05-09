import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def _batch(tree):
    if isinstance(tree, dict):
        return {key: _batch(value) for key, value in tree.items()}
    return np.stack([np.asarray(tree)], axis=0)


def test_libero_inputs_preserves_skill_labels():
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05)
    data = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the mug",
        "skill_id": np.asarray(2, dtype=np.int32),
        "skill_mask": np.asarray(True, dtype=np.bool_),
    }

    inputs = transform(data)
    observation = _model.Observation.from_dict(_batch(inputs))

    assert inputs["skill_id"] == data["skill_id"]
    assert inputs["skill_mask"] == data["skill_mask"]
    assert observation.skill_id.shape == (1,)
    assert observation.skill_mask.shape == (1,)
    assert int(observation.skill_id[0]) == 2
    assert bool(observation.skill_mask[0])
