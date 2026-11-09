"""Small JSON configuration layer; usable without any GPU dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

MODELS = {
    "qwen3_1p7b_base": "Qwen/Qwen3-1.7B-Base",
    "qwen3_4b_base": "Qwen/Qwen3-4B-Base",
    "qwen35_2b_base": "Qwen/Qwen3.5-2B-Base",
    "qwen35_4b_base": "Qwen/Qwen3.5-4B-Base",
}


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
    data = config["data"]
    kind = data.get("kind", "math")
    if kind not in {"math", "gsm8k", "deepmath"}:
        raise ValueError("data.kind must be math, gsm8k or deepmath")
    for field in ("name", "registry_name", "math_data_name"):
        if data.get(field) and not re.fullmatch(r"[A-Za-z0-9_-]+", data[field]):
            raise ValueError(f"data.{field} must be a simple name, not a path")
    size = data.get("train_size", 2048)
    if size != "all" and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise ValueError("data.train_size must be a positive integer or 'all'")
    for key, default in (("dev_size", 256), ("pilot_size", 2048), ("debug_size", 32)):
        value = data.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if kind == "math" and size != "all" and size != data.get("pilot_size", 2048):
        raise ValueError("For a MATH subset, set data.pilot_size equal to data.train_size")
    if config["model"].get("kind", "base") != "base":
        raise ValueError("This protocol uses Base models with the shared raw-completion prompt")
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
        isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= config["evaluation"]["attempts"] for k in config["evaluation"]["ks"]
    ):
        raise ValueError("Each evaluation k must satisfy 1 <= k <= attempts")
    sft = config["rft"]["sft"]
    for key in ("epochs", "batch_size", "max_length"):
        if not isinstance(sft[key], int) or sft[key] < 1:
            raise ValueError(f"rft.sft.{key} must be a positive integer")
    for label, section in (("rft.sft", sft), ("grpo", config["grpo"])):
        value = section.get("max_steps")
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{label}.max_steps must be a positive integer")
    keep = sft.get("max_ckpt_to_keep")
    if keep is not None and (not isinstance(keep, int) or isinstance(keep, bool) or keep < 1):
        raise ValueError("rft.sft.max_ckpt_to_keep must be a positive integer")
    if sft["lora_rank"] != 0:
        raise ValueError("This comparison/export path uses full fine-tuning; set lora_rank=0")
    if sft["max_length"] < config["sampling"]["max_tokens"] + config["sampling"]["max_prompt_length"] + 1:
        raise ValueError("SFT max_length must cover prompt + response budgets and the reconstructed EOS")
    if config["server"]["host"] != "127.0.0.1":
        raise ValueError("The managed inference server must use 127.0.0.1")


def select_model(config: dict, key: str) -> dict:
    """Select one model without accidentally reusing another model's output."""
    if key not in MODELS:
        raise ValueError(f"Unknown model: {key}")
    previous = config["model"].get("key")
    name = config["experiment"]["name"]
    if previous == key:
        return config
    name = name.replace(previous, key, 1) if previous and previous in name else f"{name}_{key}"
    return merge(config, {"experiment": {"name": name}, "model": {
        "key": key, "repo_id": MODELS[key], "revision": "main", "local_path": None, "kind": "base",
    }})


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
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item for item in os.environ.get(key, "").split(",") if item]
        os.environ[key] = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost"]))
