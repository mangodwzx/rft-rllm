"""Small JSON configuration layer; usable without any GPU dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: Path, seen: frozenset[Path] = frozenset()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, seen | {path}), config)
    validate(config)
    return config


def validate(config: dict) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["experiment"]["name"]):
        raise ValueError("experiment.name may contain only letters, digits, '_' and '-'")
    for section, keys in {
        "sampling": ["max_tokens", "max_prompt_length"],
        "resources": ["gpus", "tensor_parallel_size"],
        "rft": ["rounds", "attempts"],
        "grpo": ["batch_size", "group_size", "epochs"],
        "evaluation": ["attempts", "concurrency"],
    }.items():
        for key in keys:
            value = config[section][key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if config["sampling"]["temperature"] <= 0:
        raise ValueError("pass@k requires stochastic sampling: temperature must be > 0")
    if not 0 < config["sampling"]["top_p"] <= 1:
        raise ValueError("sampling.top_p must be in (0, 1]")
    if config["grpo"]["group_size"] < 2:
        raise ValueError("GRPO needs group_size >= 2")
    if config["resources"]["gpus"] % config["resources"]["tensor_parallel_size"]:
        raise ValueError("gpus must be divisible by tensor_parallel_size")
    if not config["evaluation"]["ks"] or any(
        not isinstance(k, int) or not 1 <= k <= config["evaluation"]["attempts"] for k in config["evaluation"]["ks"]
    ):
        raise ValueError("Each evaluation k must satisfy 1 <= k <= attempts")
    sft = config["rft"]["sft"]
    for key in ("epochs", "batch_size", "max_length"):
        if not isinstance(sft[key], int) or sft[key] < 1:
            raise ValueError(f"rft.sft.{key} must be a positive integer")
    if sft["lora_rank"] != 0:
        raise ValueError("This comparison/export path uses full fine-tuning; set lora_rank=0")
    if sft["max_length"] < config["sampling"]["max_tokens"] + config["sampling"]["max_prompt_length"]:
        raise ValueError("SFT max_length must cover the entire prompt + response budget")
    if config["server"]["host"] != "127.0.0.1":
        raise ValueError("The managed inference server must use 127.0.0.1")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_environment(root: Path) -> None:
    # Set before importing rllm: DatasetRegistry caches some paths at import time.
    os.environ["RLLM_HOME"] = str(root / ".rllm")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["PYTHONPATH"] = str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
