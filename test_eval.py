import os
CACHE_DIR = '/fs/nexus-projects/wilddiffusion/cache'
os.environ["OPENPI_DATA_HOME"] = CACHE_DIR

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

config = _config.get_config("pi05_libero")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

# Create a trained policy.
model = config.model.load_pytorch(config, checkpoint_dir)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
data_config = config.data.create(config.assets_dirs, config.model)
import pdb;pdb.set_trace()


base_image = _parse_image(data["observation/image"])
wrist_image = _parse_image(data["observation/wrist_image"])

example =  {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "What should do next",
    }

action_chunk = policy.infer(example)["actions"]
