import dataclasses
import logging
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """Loads a checkpoint and initializes selected missing parameters from the current model init."""

    params_path: str
    missing_regex: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class LlmSkillEmbeddingCheckpointWeightLoader(WeightLoader):
    """Loads a checkpoint and initializes skill embeddings from the LLM token embedding table."""

    params_path: str
    skill_vocab: Sequence[str]
    missing_regex: str = ".*(skill_|lora).*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        merged_params = _merge_params(loaded_params, params, missing_regex=self.missing_regex)

        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_merged = flax.traverse_util.flatten_dict(merged_params, sep="/")
        embedding_key = "PaliGemma/llm/embedder/input_embedding"
        skill_key = "skill_emb_bank"

        if embedding_key not in flat_merged:
            raise KeyError(f"Cannot initialize skill embeddings because {embedding_key!r} is missing.")
        if skill_key not in flat_ref:
            raise KeyError(f"Cannot initialize skill embeddings because {skill_key!r} is not in the target model.")

        llm_embedding = np.asarray(flat_merged[embedding_key])
        skill_shape = flat_ref[skill_key].shape
        if len(skill_shape) != 2:
            raise ValueError(f"Expected {skill_key!r} to be rank 2, got shape {skill_shape}.")
        if len(self.skill_vocab) != skill_shape[0]:
            raise ValueError(
                f"skill_vocab has {len(self.skill_vocab)} entries but {skill_key!r} expects {skill_shape[0]} rows."
            )
        if llm_embedding.shape[1] != skill_shape[1]:
            raise ValueError(
                f"LLM embedding dim {llm_embedding.shape[1]} does not match {skill_key!r} dim {skill_shape[1]}. "
                "Set skill_emb_dim to the VLM embedding width for direct LLM initialization."
            )

        tokenizer = _tokenizer.PaligemmaTokenizer(max_len=16)
        sp_tokenizer = tokenizer._tokenizer  # Reuse the exact tokenizer used for PaliGemma prompts.
        # Keep only the *direction* of each skill's LLM embedding and rescale it
        # to the from-scratch skill_emb_bank magnitude (init = normal * 0.02, so
        # row norm ~= 0.02 * sqrt(dim)). The raw LLM embeddings live at
        # transformer-hidden scale (row norm ~tens); feeding that into the
        # freshly-initialized skill conditioning heads (skill_to_adarms /
        # skill_to_action_film) injects a ~100x oversized signal and blows up
        # the action loss. Matching the small from-scratch scale preserves the
        # LLM semantics (direction) without the scale shock.
        target_norm = 0.02 * np.sqrt(skill_shape[1])
        skill_rows = []
        for skill in self.skill_vocab:
            cleaned_skill = skill.strip().replace("_", " ").replace("\n", " ")
            token_ids = sp_tokenizer.encode(cleaned_skill, add_bos=False)
            if not token_ids:
                raise ValueError(f"Skill {skill!r} produced no PaliGemma tokens.")
            row = np.mean(llm_embedding[np.asarray(token_ids, dtype=np.int32)], axis=0)
            row = row / (np.linalg.norm(row) + 1e-8) * target_norm
            skill_rows.append(row)

        flat_merged[skill_key] = np.stack(skill_rows, axis=0).astype(flat_ref[skill_key].dtype)
        return flax.traverse_util.unflatten_dict(flat_merged, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
