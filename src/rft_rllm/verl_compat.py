"""Narrow compatibility adapter for the pinned rLLM -> verl tensor boundary.

rLLM's pinned batch builder identifies padding by token value. Base tokenizers
can use EOS as their pad ID, so that rule removes real terminal EOS tokens.
Restore attention/position masks from the captured sequence lengths instead.
The official token tensors, response mask, rewards, advantages and loss stay
unchanged. This module is also a Ray worker_process_setup_hook.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import textwrap

from . import VERL_VERSION

BUILDER_SHA256 = "b8459a94dea60b47448f516c3915ec165a7e5ce518a7772087b449220c599f87"
ADAPTER_NAME = "rllm_pinned_length_attention_mask_v1"


def restore_length_masks(result, accumulated, max_prompt_length: int, max_response_length: int, processor, multimodal_handler):
    """Restore only attention_mask and position_ids on the official DataProto."""
    import torch

    inputs = result.batch["input_ids"]
    rows = len(accumulated.prompts)
    if len(accumulated.responses) != rows or inputs.shape != (rows, max_prompt_length + max_response_length):
        raise ValueError("Unexpected official batch shape; refusing to apply the length-mask adapter")
    if result.batch["response_mask"].shape != (rows, max_response_length):
        raise ValueError("Unexpected official response mask shape")
    prompt_lengths = torch.tensor([min(len(ids), max_prompt_length) for ids in accumulated.prompts], device=inputs.device)
    response_lengths = torch.tensor([min(len(ids), max_response_length) for ids in accumulated.responses], device=inputs.device)
    if (prompt_lengths < 1).any() or (response_lengths < 1).any():
        raise ValueError("Expected nonempty captured prompt and response IDs")
    prompt_mask = torch.arange(max_prompt_length, device=inputs.device)[None, :] >= (max_prompt_length - prompt_lengths[:, None])
    response_mask = torch.arange(max_response_length, device=inputs.device)[None, :] < response_lengths[:, None]
    attention_mask = torch.cat((prompt_mask, response_mask), dim=1)
    if processor is not None and hasattr(processor, "image_processor") and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        positions = multimodal_handler(processor, inputs, attention_mask, accumulated.multi_modal_inputs)
    else:
        positions = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
    result.batch["attention_mask"] = attention_mask
    result.batch["position_ids"] = positions
    result.meta_info["rft_attention_mask_adapter"] = ADAPTER_NAME
    return result


def install() -> None:
    """Install once per process, failing closed on another upstream version."""
    if importlib.metadata.version("verl") != VERL_VERSION:
        raise RuntimeError(f"The length-mask adapter requires verl=={VERL_VERSION}")
    transform = importlib.import_module("rllm.trainer.verl.transform")
    original = transform._batch_tensors_and_build_data_proto
    if getattr(original, "_rft_length_mask_adapter", None) == ADAPTER_NAME:
        return
    actual = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).strip().encode()).hexdigest()
    if actual != BUILDER_SHA256:
        raise RuntimeError("Pinned rLLM batch-builder source changed; review the length-mask adapter before training")

    @functools.wraps(original)
    def build(accumulated, pad_token_id, max_prompt_length, max_response_length, processor=None):
        result = original(accumulated, pad_token_id, max_prompt_length, max_response_length, processor)
        return restore_length_masks(
            result, accumulated, max_prompt_length, max_response_length, processor,
            transform._handle_multimodal_position_ids,
        )

    build._rft_length_mask_adapter = ADAPTER_NAME
    transform._batch_tensors_and_build_data_proto = build


def initialize_ray(config):
    """Reuse the official Ray settings and add the adapter in every worker.

    The pinned launcher reads top-level ray_init via get_ray_init_settings; it
    does not forward ray_kwargs.ray_init. Initializing here prevents a driver-only
    patch from silently missing the VerlTaskRunner actor that builds PPO batches.
    """
    import ray
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    if ray.is_initialized():
        raise RuntimeError("Launch GRPO in a fresh Python process so the Ray length-mask worker hook is guaranteed")
    runtime_env = get_ppo_ray_runtime_env()
    runtime_env["worker_process_setup_hook"] = "rft_rllm.verl_compat.install"
    ray.init(runtime_env=runtime_env, **get_ray_init_settings(config))
    return ray
