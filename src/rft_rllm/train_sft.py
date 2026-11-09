"""Official rLLM/verl SFT with a raw-completion data adapter for Base models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prompts import RAW_CHAT_TEMPLATE


def sft_overrides(settings: dict, train_count: int | None = None) -> dict:
    overrides = {
        "model": {"custom_chat_template": RAW_CHAT_TEMPLATE},
        "data": {
            "custom_cls": {"path": "pkg://rft_rllm.raw_sft_dataset", "name": "RawCompletionSFTDataset"},
            "pad_mode": "no_padding",
            "truncation": "error",
        },
        "trainer": {"n_gpus_per_node": settings["resources"]["gpus"], "resume_mode": "disable"},
    }
    sft = settings["rft"]["sft"]
    if sft.get("max_ckpt_to_keep") is not None:
        overrides["trainer"]["max_ckpt_to_keep"] = int(sft["max_ckpt_to_keep"])
    if sft.get("max_steps") is not None:
        max_steps = int(sft["max_steps"])
        if train_count is not None:
            available_steps = train_count // int(sft["batch_size"]) * int(sft["epochs"])
            # verl saves the final checkpoint at total_training_steps; a cap
            # beyond the available full batches would miss that final save.
            max_steps = min(max_steps, available_steps)
        if max_steps < 1:
            raise ValueError("SFT step limit requires at least one full training batch and a positive max_steps")
        overrides["trainer"]["total_training_steps"] = max_steps
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty SFT output: {output}")

    from rllm.data import DatasetRegistry
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft import SFTSpec
    from transformers import AutoTokenizer

    dataset = DatasetRegistry.load_dataset(args.dataset, "train")
    if dataset is None or len(dataset) < settings["rft"]["sft"]["batch_size"]:
        raise ValueError("Prepare at least one full batch of accepted trajectories before SFT")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template != RAW_CHAT_TEMPLATE:
        raise ValueError("The model must use the shared identity template; run prepare/freeze first")
    sft = settings["rft"]["sft"]
    spec = SFTSpec(
        model=args.model, train_dataset=dataset, val_dataset=None,
        lr=float(sft["lr"]), lr_schedule="constant", epochs=int(sft["epochs"]),
        batch_size=int(sft["batch_size"]), max_length=int(sft["max_length"]),
        lora_rank=0, tokenize_method="cumulative", val_freq=-1,
        project="rft-rllm", experiment=settings["experiment"]["name"],
        output_dir=str(output), overrides=sft_overrides(settings, train_count=len(dataset)),
    )
    AgentSFTTrainer(spec, backend="verl").train()


if __name__ == "__main__":
    main()
