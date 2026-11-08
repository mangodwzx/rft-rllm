"""Frozen, auditable MATH/GSM8K/DeepMath data protocols.

MATH pilot and full training share one held-out development set. Registry names
are unique per physical split because the pinned rLLM eval caches materialized
benchmarks by name rather than by (name, split).
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import write_json

SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
REFERENCE_CORRECTIONS = Path(__file__).with_name("reference_corrections.json")
SCHEMA_VERSION = 2


def _digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _key(question: str) -> str:
    return _digest(" ".join(unicodedata.normalize("NFKC", question).split()))


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _name(config: dict) -> str:
    cfg = config["data"]
    name = cfg.get("name") or cfg.get("protocol_id") or cfg.get("kind", "math")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("data.name/protocol_id may contain only letters, digits, '_' and '-'")
    return name


def data_directory(config: dict, root: Path) -> Path:
    return Path(root).resolve() / "data" / _name(config)


def _physical_split(config: dict, split: str) -> str:
    cfg = config["data"]
    if cfg.get("kind", "math") == "math" and split == "train":
        if cfg.get("train_size", 2048) == "all":
            return "full"
        if cfg.get("train_size", 2048) != cfg.get("pilot_size", 2048):
            raise ValueError("MATH train_size must equal pilot_size or 'all'; use a new data.name for another pilot size")
        return "pilot"
    return split


def _effective_config(config: dict) -> dict:
    """Canonical effective data options, excluding irrelevant preset inheritance."""
    cfg = config["data"]
    kind = cfg.get("kind", "math")
    defaults = {"math": "EleutherAI/hendrycks_math", "gsm8k": "openai/gsm8k", "deepmath": "zwhe99/DeepMath-103K"}
    if kind not in defaults:
        raise ValueError(f"Unknown data.kind: {kind!r}")
    effective = {
        "kind": kind, "name": _name(config), "seed": cfg.get("seed", 42),
        "source": cfg.get("source", defaults[kind]), "source_revision": cfg.get("source_revision") or "main",
        "source_split": cfg.get("source_split", "train"),
    }
    if cfg.get("train_jsonl"):
        effective["train_jsonl"] = cfg["train_jsonl"]
    if kind == "math":
        effective.update(
            pilot_size=cfg.get("pilot_size", 2048), dev_size=cfg.get("dev_size", 256),
            subjects=cfg.get("subjects", list(SUBJECTS)),
            test_source=cfg.get("test_source", "HuggingFaceH4/MATH-500"),
            test_revision=cfg.get("test_revision") or "main",
        )
        if cfg.get("test_jsonl"):
            effective["test_jsonl"] = cfg["test_jsonl"]
    else:
        effective["source_config"] = cfg.get("source_config", "main" if kind == "gsm8k" else None)
        if kind == "gsm8k":
            effective["debug_size"] = cfg.get("debug_size", 32)
        else:
            effective.update(train_size=cfg.get("train_size", 8192), dev_size=cfg.get("dev_size", 256),
                             math_data_name=cfg.get("math_data_name", "math"))
            effective["math"] = _effective_config(_math_config(config))
    return effective


def registry_name(config: dict, split: str) -> str:
    cfg = _effective_config(config)
    prefix = config["data"].get("registry_name") or _name(config)
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("data.registry_name must be a simple registry identifier")
    fingerprint = _digest(json.dumps(cfg, sort_keys=True, ensure_ascii=False))[:12]
    return f"{prefix}_{fingerprint}_{_physical_split(config, split)}"


def _boxed(text: str) -> str | None:
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\b", text))
    if not boxes:
        return None
    suffix = text[boxes[-1].end():]
    opening = re.match(r"\s*\{", suffix)
    if not opening:
        return None
    start, depth = boxes[-1].end() + opening.end(), 1
    for index in range(start, len(text)):
        backslashes, cursor = 0, index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes, cursor = backslashes + 1, cursor - 1
        if backslashes % 2:
            continue
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index].strip()
    return None


@lru_cache(maxsize=1)
def _reference_corrections() -> dict[str, dict]:
    entries = json.loads(REFERENCE_CORRECTIONS.read_text(encoding="utf-8"))
    return {entry["problem_sha256"]: entry for entry in entries}


def math_reference_answer(question: str, solution: str) -> tuple[str | None, dict | None]:
    """Repair source annotations only; model answer grading remains strict."""
    answer = _boxed(solution)
    if answer:
        return answer, None
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\b", solution))
    if boxes:
        digit = re.match(r"\s+([0-9])\s*\$", solution[boxes[-1].end():])
        if digit:
            return digit[1], {"kind": "unbraced_digit_box", "solution_sha256": _digest(solution)}
    correction = _reference_corrections().get(_key(question))
    if correction and _digest(solution) == correction["solution_sha256"]:
        return correction["answer"], {"kind": "audited_empty_box", **correction}
    return None, None


def _normalize(raw: dict[str, Any], source: str) -> dict[str, Any]:
    question = raw.get("question") or raw.get("problem")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each dataset row needs a non-empty question/problem")
    adjustment = None
    difficulty = None
    level = None
    if source == "gsm8k":
        solution = str(raw.get("solution") or raw.get("answer") or "")
        answer = raw.get("ground_truth")
        if answer is None:
            if "####" not in solution:
                raise ValueError("GSM8K reference answer has no #### delimiter")
            answer = solution.rsplit("####", 1)[1].strip().replace(",", "")
        subject = "arithmetic"
    elif source == "deepmath":
        solution = str(raw.get("solution") or raw.get("r1_solution_1") or "")
        answer = raw.get("ground_truth", raw.get("final_answer"))
        subject = str(raw.get("subject") or raw.get("topic") or "unknown")
        if raw.get("difficulty") is not None:
            difficulty = float(raw["difficulty"])
            if not math.isfinite(difficulty):
                raise ValueError("DeepMath difficulty must be finite")
    else:
        solution = str(raw.get("solution") or "")
        answer = raw.get("ground_truth", raw.get("answer"))
        if answer is None:
            if source in {"train", "math"}:
                answer, adjustment = math_reference_answer(question, solution)
            else:
                answer = _boxed(solution)
        level = raw.get("level")
        if level is not None:
            if str(level).strip() == "Level ?" and source in {"train", "math"}:
                level = None
            else:
                match = re.fullmatch(r"(?:Level\s*)?([1-5])", str(level).strip(), re.IGNORECASE)
                if not match:
                    raise ValueError(f"Invalid MATH level: {level!r}")
                level = int(match.group(1))
        subject = str(raw.get("subject") or raw.get("type") or "unknown")
    if answer is None or not str(answer).strip():
        raise ValueError(f"Missing mathematical reference answer: {question[:100]!r}")
    source_id = raw.get("id")
    if source_id is None:
        source_id = raw.get("unique_id") or _key(question)
    return {
        "id": f"{source}:{source_id}", "question": question.strip(),
        "ground_truth": str(answer).strip(), "solution": solution,
        "subject": subject, "level": level, "difficulty": difficulty,
        "problem_sha256": _key(question), "reference_answer_adjustment": adjustment,
    }


def _deduplicate(rows: list[dict], excluded: set[str]) -> tuple[list[dict], list[dict]]:
    unique, identities, removed = {}, {}, []
    for row in sorted(rows, key=lambda item: item["id"]):
        key = _key(row["question"])
        if row["id"] in identities and identities[row["id"]] != key:
            raise ValueError(f"Dataset ID refers to different questions: {row['id']}")
        identities[row["id"]] = key
        if key in excluded:
            removed.append({"id": row["id"], "problem_sha256": key, "reason": "protected_split_overlap"})
        elif key in unique:
            if row["ground_truth"] != unique[key]["ground_truth"]:
                raise ValueError(f"Duplicate question has conflicting answers: {row['id']}")
            removed.append({"id": row["id"], "problem_sha256": key, "reason": "duplicate_question"})
        else:
            unique[key] = row
    return list(unique.values()), removed


def stratified_take(rows: list[dict], count: int, seed: int, salt: str) -> tuple[list[dict], list[dict]]:
    """Exact proportional subject x level allocation with stable hash ordering."""
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(rows):
        raise ValueError(f"Cannot select {count!r} rows from {len(rows)}")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["subject"], str(row["level"]))].append(row)
    keys = sorted(groups)
    # Integer remainders avoid floating-point tie-breaking differences.
    allocated = {key: count * len(groups[key]) // len(rows) for key in keys}
    remainder = count - sum(allocated.values())
    ranked_keys = sorted(keys, key=lambda key: (-(count * len(groups[key]) % len(rows)), key))
    for key in ranked_keys[:remainder]:
        allocated[key] += 1
    selected, remaining = [], []
    for key in keys:
        ranked = sorted(groups[key], key=lambda row: _digest(f"{seed}:{salt}:{_key(row['question'])}"))
        selected.extend(ranked[:allocated[key]])
        remaining.extend(ranked[allocated[key]:])
    order = lambda row: _digest(f"{seed}:{salt}:order:{_key(row['question'])}")
    return sorted(selected, key=order), sorted(remaining, key=order)


def _source(cfg: dict, root: Path, *, test: bool) -> tuple[list[dict], dict]:
    local = cfg.get("test_jsonl" if test else "train_jsonl")
    if local:
        path = (root / local).resolve()
        return _read(path), {"kind": "local", "path": str(path), "sha256": _digest(path.read_bytes())}
    from datasets import load_dataset
    from huggingface_hub import HfApi

    kind = cfg.get("kind", "math")
    defaults = {"math": "EleutherAI/hendrycks_math", "gsm8k": "openai/gsm8k", "deepmath": "zwhe99/DeepMath-103K"}
    repo = cfg.get("test_source", "HuggingFaceH4/MATH-500") if test else cfg.get("source", defaults[kind])
    requested = cfg.get("test_revision" if test else "source_revision") or "main"
    revision = HfApi().dataset_info(repo, revision=requested).sha
    if not revision:
        raise RuntimeError(f"Cannot resolve dataset revision: {repo}@{requested}")
    names = [None] if test else (cfg.get("subjects", list(SUBJECTS)) if kind == "math" else [cfg.get("source_config", "main" if kind == "gsm8k" else None)])
    if not names:
        raise ValueError("data.subjects must not be empty")
    split = "test" if test else cfg.get("source_split", "train")
    if kind in {"math", "gsm8k", "deepmath"} and not test and split != "train":
        raise ValueError("Training/debug source must use the official train split")
    rows = []
    for name in names:
        rows.extend(dict(row) for row in load_dataset(repo, name, split=split, revision=revision))
    return rows, {"kind": "huggingface", "repo": repo, "revision": revision,
                  "requested_revision": requested, "configs": names, "split": split}


def _demonstration_keys() -> set[str]:
    # Prompts are lightweight and contain no training framework imports.
    from .prompts import DEMONSTRATIONS
    return {_key(item[0] if not isinstance(item, dict) else item["question"]) for item in DEMONSTRATIONS}


def _requested(config: dict, root: Path) -> dict:
    cfg = _effective_config(config)

    def fingerprint_local(options: dict) -> None:
        for key in ("train_jsonl", "test_jsonl"):
            if options.get(key):
                path = (root / options[key]).resolve()
                options[key] = {"path": str(path), "sha256": _digest(path.read_bytes())}
        if isinstance(options.get("math"), dict):
            fingerprint_local(options["math"])

    fingerprint_local(cfg)
    cfg["reference_corrections_sha256"] = _digest(REFERENCE_CORRECTIONS.read_bytes())
    cfg["demonstration_question_keys"] = sorted(_demonstration_keys())
    return cfg


def _verify_manifest(directory: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported data manifest schema; use a new data.name")
    for split, info in manifest["splits"].items():
        path = directory / info["file"]
        if not path.is_file() or _digest(path.read_bytes()) != info["sha256"]:
            raise ValueError(f"Prepared {split} data differs from its manifest")


def load_rows(config: dict, root: Path, split: str) -> list[dict]:
    directory = data_directory(config, root)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    physical = _physical_split(config, split)
    if physical not in manifest["splits"]:
        raise ValueError(f"Dataset {_name(config)!r} has no {split!r} split")
    info = manifest["splits"][physical]
    path = directory / info["file"]
    if not path.is_file() or _digest(path.read_bytes()) != info["sha256"]:
        raise ValueError(f"Prepared {split} data differs from its manifest")
    return _read(path)


def _register(config: dict, root: Path, manifest: dict) -> None:
    from rllm.data.dataset import DatasetRegistry
    for split in manifest["splits"]:
        # Only public active splits, not the unused MATH training scale.
        if split not in {"train", "dev", "test", "debug"}:
            continue
        DatasetRegistry.register_dataset(name=registry_name(config, split), data=load_rows(config, root, split),
                                         split=split, category="math")


def _view(config: dict, manifest: dict) -> dict:
    result = copy.deepcopy(manifest)
    if config["data"].get("kind", "math") == "math":
        physical = _physical_split(config, "train")
        result["splits"]["train"] = {**result["splits"][physical], "physical_split": physical}
    result["registry_names"] = {split: registry_name(config, split) for split in result["splits"]
                                if split in {"train", "dev", "test", "debug"}}
    return result


def _audit(rows: list[dict]) -> dict:
    return {
        "reference_answer_adjustments": [
            {"id": row["id"], "problem_sha256": _key(row["question"]), "answer": row["ground_truth"],
             **row["reference_answer_adjustment"]}
            for row in rows if row.get("reference_answer_adjustment")
        ],
        "unknown_training_levels": [row["id"] for row in rows if row["level"] is None],
    }


def _math_config(config: dict) -> dict:
    cfg = config["data"]
    math_cfg = {"name": cfg.get("math_data_name", "math"), "kind": "math", "train_size": "all",
                "pilot_size": 2048, "dev_size": cfg.get("dev_size", 256), "seed": cfg.get("seed", 42)}
    math_cfg.update(cfg.get("math", {}))
    if math_cfg.get("kind") != "math":
        raise ValueError("data.math must describe a MATH protocol")
    if math_cfg["name"] != cfg.get("math_data_name", "math"):
        raise ValueError("data.math.name must match data.math_data_name")
    return {"data": math_cfg}


def _math_dependency(config: dict, root: Path) -> tuple[dict, dict]:
    math_config = _math_config(config)
    return math_config, prepare(math_config, root, register=False)


def _verify_protected_math(config: dict, root: Path, manifest: dict) -> None:
    _, current = _math_dependency(config, root)
    recorded = manifest["sources"]["protected_math"]
    if any(recorded[f"{split}_sha256"] != current["splits"][split]["sha256"] for split in ("dev", "test")):
        raise ValueError("DeepMath protected MATH dev/test hashes changed; use a new data.name")


def prepare(config: dict, root: Path, register: bool = True) -> dict:
    """Prepare immutable files; changing MATH train_size alone reuses the split."""
    cfg = config["data"]
    kind = cfg.get("kind", "math")
    if kind not in {"math", "gsm8k", "deepmath"}:
        raise ValueError(f"Unknown data.kind: {kind!r}")
    root = Path(root).resolve()
    output = data_directory(config, root)
    requested = _requested(config, root)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("requested") != requested:
            raise ValueError("Existing data manifest differs from the requested protocol; use a new data.name")
        _verify_manifest(output, manifest)
        if kind == "deepmath":
            _verify_protected_math(config, root, manifest)
        result = _view(config, manifest)
        if register:
            _register(config, root, result)
        return result
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Non-empty data directory without a manifest; use a new data.name")
    seed = cfg.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("data.seed must be an integer")
    protected = _demonstration_keys()
    sources, removed, audit = {}, {}, {}
    if kind == "math":
        _physical_split(config, "train")
        raw_test, sources["test"] = _source(cfg, root, test=True)
        test, removed["test"] = _deduplicate([_normalize(row, "test") for row in raw_test], set())
        if not test or removed["test"]:
            raise ValueError("Held-out MATH-500 must be nonempty and contain no duplicate questions")
        if not cfg.get("test_jsonl") and len(test) != 500:
            raise ValueError(f"Expected official MATH-500's 500 rows, got {len(test)}")
        protected.update(_key(row["question"]) for row in test)
        raw_train, sources["train"] = _source(cfg, root, test=False)
        training = [_normalize(row, "train") for row in raw_train]
        audit = _audit(training)
        training, removed["train"] = _deduplicate(training, protected)
        dev, full = stratified_take(training, cfg.get("dev_size", 256), seed, "development")
        pilot, _ = stratified_take(full, cfg.get("pilot_size", 2048), seed, "pilot")
        rows = {"pilot": pilot, "full": full, "dev": dev, "test": test}
        method = "subject x level proportional largest-remainder sampling, stable question hashes; dev before pilot"
    elif kind == "gsm8k":
        raw, sources["debug"] = _source(cfg, root, test=False)
        candidates, removed["debug"] = _deduplicate([_normalize(row, "gsm8k") for row in raw], protected)
        size = cfg.get("debug_size", 32)
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= len(candidates):
            raise ValueError("debug_size must be positive and fit the unique GSM8K train rows")
        candidates.sort(key=lambda row: _digest(f"{seed}:gsm8k:{_key(row['question'])}"))
        rows = {"debug": candidates[:size]}
        method = "fixed hash-ranked GSM8K train subset; generation/grading debug only"
    else:
        size = cfg.get("train_size", 8192)
        if isinstance(size, bool) or size not in {8192, 16384}:
            raise ValueError("DeepMath train_size must be 8192 or 16384")
        math_config, math_manifest = _math_dependency(config, root)
        dev, test = load_rows(math_config, root, "dev"), load_rows(math_config, root, "test")
        protected.update(_key(row["question"]) for row in dev + test)
        sources["protected_math"] = {"directory": str(data_directory(math_config, root)),
                                      "dev_sha256": math_manifest["splits"]["dev"]["sha256"],
                                      "test_sha256": math_manifest["splits"]["test"]["sha256"]}
        raw, sources["train"] = _source(cfg, root, test=False)
        candidates, removed["train"] = _deduplicate([_normalize(row, "deepmath") for row in raw], protected)
        if len(candidates) < size:
            raise ValueError(f"Need {size} non-overlapping DeepMath rows, found {len(candidates)}")
        candidates.sort(key=lambda row: _digest(f"{seed}:deepmath:{_key(row['question'])}"))
        rows = {"train": candidates[:size], "dev": dev, "test": test}
        method = "fixed hash-ranked DeepMath train prefix; excludes MATH dev/test and demonstrations"
    manifest = {
        "schema_version": SCHEMA_VERSION, "requested": requested, "kind": kind,
        "seed": seed, "split_method": method, "sources": sources, "removed_rows": removed,
        "deduplication": "NFKC and whitespace-normalized exact question hash; no semantic-overlap guarantee",
        "reference_corrections_sha256": _digest(REFERENCE_CORRECTIONS.read_bytes()),
        "demonstration_question_keys": sorted(_demonstration_keys()), **audit, "splits": {},
    }
    if kind == "math":
        manifest.update(full_train_includes_pilot=True, full_train_excludes_dev=True)
    output.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        path = output / f"{split}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items).encode("utf-8")
        path.write_bytes(payload)
        strata = collections.Counter(f"{row['subject']}|{row['level']}" for row in items)
        manifest["splits"][split] = {"count": len(items), "file": path.name, "sha256": _digest(payload),
                                      "strata": dict(sorted(strata.items()))}
    write_json(manifest_path, manifest)
    result = _view(config, manifest)
    if register:
        _register(config, root, result)
    return result
