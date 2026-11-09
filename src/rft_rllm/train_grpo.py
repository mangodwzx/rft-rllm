"""Launch rLLM's official unified GRPO trainer with the local verl backend.

The pinned rLLM ``train`` CLI selects Tinker, so this entry point follows its
official ``cookbooks/math/train.py`` Python recipe. No loss or trainer is
implemented here. Heavy imports are delayed until execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prompts import RAW_CHAT_TEMPLATE, STOP_STRINGS


def _override(key: str, value: Any) -> str:
    """Encode one Hydra value without shell interpolation."""
    return f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def build_overrides(settings: dict[str, Any], model: str, output: Path) -> list[str]:
    """Map experiment settings to the pinned upstream configuration schema."""
    grpo = settings["grpo"]
    sampling = settings["sampling"]
    resources = settings["resources"]
    batch_size = int(grpo["batch_size"])
    group_size = int(grpo["group_size"])
    gpus = int(resources["gpus"])
    tp = int(resources["tensor_parallel_size"])
    if batch_size < 1 or group_size < 2 or gpus < 1 or tp < 1 or gpus % tp:
        raise ValueError("Need batch_size >= 1, group_size >= 2, and gpus divisible by tensor_parallel_size.")
    max_prompt = int(sampling["max_prompt_length"])
    max_response = int(sampling["max_tokens"])
    if min(max_prompt, max_response, int(grpo["epochs"])) < 1:
        raise ValueError("Prompt/response lengths and epochs must be positive.")

    values: dict[str, Any] = {
        "+model.name": model,
        "actor_rollout_ref.model.path": model,
        "actor_rollout_ref.model.custom_chat_template": RAW_CHAT_TEMPLATE,
        "actor_rollout_ref.model.lora.rank": 0,
        "actor_rollout_ref.actor.optim.lr": float(grpo["lr"]),
        "actor_rollout_ref.actor.data_loader_seed": int(grpo["seed"]),
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch_size,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": max(8192, max_prompt + max_response),
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.seed": int(grpo["seed"]),
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": tp,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.max_model_len": max_prompt + max_response,
        "actor_rollout_ref.rollout.gpu_memory_utilization": float(resources.get("gpu_memory_utilization", 0.5)),
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template": RAW_CHAT_TEMPLATE,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template_content_format": "string",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config": "vllm",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "rllm.data.train_batch_size": batch_size,
        "rllm.data.max_prompt_length": max_prompt,
        "rllm.data.max_response_length": max_response,
        "rllm.data.seed": int(grpo["seed"]),
        "rllm.workflow.retry_limit": 1,
        "rllm.algorithm.adv_estimator": "grpo",
        "rllm.algorithm.norm_adv_by_std_in_grpo": True,
        "rllm.algorithm.loss_agg_mode": "seq-mean-token-mean",
        "rllm.algorithm.kl_beta": 0.0,
        "rllm.rollout.n": group_size,
        "rllm.rollout.n_val": 1,
        "rllm.rollout.train.temperature": float(sampling["temperature"]),
        "rllm.rollout.train.top_p": float(sampling["top_p"]),
        "rllm.rollout.train.max_tokens": max_response,
        "rllm.rollout.val.temperature": float(sampling["temperature"]),
        "rllm.rollout.val.top_p": float(sampling["top_p"]),
        "rllm.rollout.val.max_tokens": max_response,
        "rllm.trainer.total_epochs": int(grpo["epochs"]),
        "rllm.trainer.test_freq": int(grpo.get("val_freq", 10)),
        "rllm.trainer.val_before_train": False,
        "rllm.trainer.project_name": "rft-rllm",
        "rllm.trainer.experiment_name": output.name,
        "rllm.trainer.logger": ["console"],
        "rllm.episode_logging.log_episodes": True,
        "rllm.episode_logging.episode_log_dir": str(output / "rollouts"),
        "trainer.default_local_dir": str(output),
        "trainer.default_hdfs_dir": None,
        "trainer.n_gpus_per_node": gpus,
        "trainer.nnodes": 1,
        "trainer.resume_mode": "disable",
        "trainer.max_actor_ckpt_to_keep": 2,
    }
    # Leave rejection sampling, uniform-group filtering and async training at
    # their official disabled defaults. Rollout mode=async above only selects
    # the vLLM server implementation; it does not enable stale-policy training.
    if grpo.get("max_steps") is not None:
        if int(grpo["max_steps"]) < 1:
            raise ValueError("grpo.max_steps must be positive when set.")
        values["rllm.trainer.total_batches"] = int(grpo["max_steps"])
    if grpo.get("save_freq") is not None:
        values["rllm.trainer.save_freq"] = int(grpo["save_freq"])
    extra = grpo.get("overrides", [])
    if not isinstance(extra, list) or not all(isinstance(value, str) for value in extra):
        raise ValueError("grpo.overrides must be a list of Hydra key=value strings.")
    return ["rllm/backend=verl", *(_override(key, value) for key, value in values.items()), *extra]


def build_config(settings: dict[str, Any], model: str, output: Path, train_count: int):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from rllm.trainer.verl.utils import sync_config

    overrides = build_overrides(settings, model, output)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        config = compose(config_name="unified", overrides=overrides)
    # Preserve upstream shared-key precedence when launched through this
    # argparse entry point instead of an @hydra.main function.
    sync_config(config, hydra_overrides=overrides)

    for mode in ("train", "val"):
        params = config.rllm.rollout[mode]
        if "stop" in params and params.stop != STOP_STRINGS:
            raise ValueError(f"rllm.rollout.{mode}.stop must match the shared raw-completion stop strings")
    if config.rllm.async_training.enable or config.rllm.rejection_sample.enable:
        raise ValueError("This baseline uses synchronous GRPO without rejection resampling.")
    if config.rllm.algorithm.adv_estimator != "grpo":
        raise ValueError("The GRPO entry point requires adv_estimator=grpo.")
    if int(config.actor_rollout_ref.model.lora.rank) != 0:
        raise ValueError("This baseline uses full fine-tuning for both RFT and GRPO; LoRA needs a separate export path.")
    batch_size = int(config.rllm.data.train_batch_size)
    if batch_size < 1 or int(config.rllm.trainer.total_epochs) < 1:
        raise ValueError("Effective GRPO batch size and epochs must be positive.")
    steps_per_epoch = train_count // batch_size
    if steps_per_epoch < 1:
        raise ValueError(f"Training data has {train_count} tasks but batch_size={batch_size}; rLLM drops incomplete batches.")
    total_steps = steps_per_epoch * int(config.rllm.trainer.total_epochs)
    cap = int(config.rllm.trainer.get("total_batches") or -1)
    if cap > 0:
        # Upstream iterates total_epochs; total_batches is an early-stop cap.
        total_steps = min(total_steps, cap)
    explicitly_set_save = settings["grpo"].get("save_freq") is not None or any(
        value.lstrip("+").split("=", 1)[0] in {"rllm.trainer.save_freq", "trainer.save_freq"}
        for value in settings["grpo"].get("overrides", [])
    )
    save_freq = int(config.rllm.trainer.save_freq) if explicitly_set_save else total_steps
    if save_freq < 1 or total_steps % save_freq:
        raise ValueError(
            f"save_freq must divide the planned {total_steps} updates: this rLLM version saves only on periodic steps."
        )
    config.rllm.trainer.save_freq = save_freq
    config.trainer.save_freq = save_freq
    OmegaConf.resolve(config)
    return config, {
        "train_tasks": train_count, "steps_per_epoch": steps_per_epoch,
        "planned_trainer_steps": total_steps, "train_batch_size": batch_size,
        "group_size": int(config.rllm.rollout.n), "epochs": int(config.rllm.trainer.total_epochs),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Experiment JSON config.")
    parser.add_argument("--model", required=True, help="Same frozen initial checkpoint used by the RFT arm.")
    parser.add_argument("--output", required=True, type=Path, help="New GRPO checkpoint directory.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config after data preparation; do not train or write files.")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = args.output.resolve()

    from omegaconf import OmegaConf

    from rllm.data import DatasetRegistry

    from rft_rllm.data import registry_name

    name = registry_name(settings, "train")
    train_dataset = DatasetRegistry.load_dataset(name, "train")
    val_dataset = DatasetRegistry.load_dataset(registry_name(settings, "dev"), "dev")
    if train_dataset is None or val_dataset is None or len(val_dataset) == 0:
        raise RuntimeError(f"Prepare the fixed {name!r} train/dev datasets before launching GRPO.")
    config, budget = build_config(settings, args.model, output, len(train_dataset))
    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(json.dumps(budget, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty GRPO output: {output}. Choose a new run directory.")
    from rft_rllm.verl_compat import ADAPTER_NAME, BUILDER_SHA256, initialize_ray, install

    install()
    budget["attention_mask_adapter"] = {"name": ADAPTER_NAME, "upstream_builder_sha256": BUILDER_SHA256}
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    (output / "budget_plan.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")

    from rllm.trainer import AgentTrainer

    from rft_rllm.flow import math_evaluator, math_flow

    trainer = AgentTrainer(
        backend="verl",
        agent_flow=math_flow,
        evaluator=math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    ray = initialize_ray(config)
    try:
        trainer.train()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
