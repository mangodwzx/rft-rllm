"""Deterministic mathematical datasets for the official rLLM registry."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
SPLITS = ("train", "dev", "test")


def _digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _key(question: str) -> str:
    return _digest(" ".join(unicodedata.normalize("NFKC", question).split()))


def _boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not starts:
        return None
    start, depth = starts[-1].end(), 1
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


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _normalize(raw: dict[str, Any], source: str) -> dict[str, Any]:
    question = raw.get("question") or raw.get("problem")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each dataset row needs a non-empty question/problem")
    solution = str(raw.get("solution") or "")
    answer = raw.get("ground_truth")
    if answer is None:
        answer = raw.get("answer")
    if answer is None:
        answer = _boxed(solution)
    if answer is None or not str(answer).strip():
        raise ValueError(f"Missing mathematical reference answer: {question[:100]!r}")
    level = raw.get("level")
    if level is not None:
        match = re.fullmatch(r"(?:Level\s*)?([1-5])", str(level).strip(), re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid MATH level: {level!r}")
        level = int(match.group(1))
    source_id = raw.get("id") or raw.get("unique_id") or _key(question)
    return {"id": f"{source}:{source_id}", "question": question.strip(),
            "ground_truth": str(answer).strip(), "solution": solution,
            "subject": str(raw.get("subject") or raw.get("type") or "unknown"), "level": level}


def _deduplicate(rows: list[dict], excluded: set[str]) -> tuple[list[dict], int]:
    unique, identities, removed = {}, {}, 0
    for row in sorted(rows, key=lambda item: item["id"]):
        key = _key(row["question"])
        if row["id"] in identities and identities[row["id"]] != key:
            raise ValueError(f"Dataset ID refers to different questions: {row['id']}")
        identities[row["id"]] = key
        if key in excluded:
            removed += 1
        elif key in unique:
            if row["ground_truth"] != unique[key]["ground_truth"]:
                raise ValueError(f"Duplicate question has conflicting answers: {row['id']}")
            removed += 1
        else:
            unique[key] = row
    return list(unique.values()), removed


def _source(cfg: dict, root: Path, *, test: bool) -> tuple[list[dict], dict]:
    local = cfg.get("test_jsonl" if test else "train_jsonl")
    if local:
        path = (root / local).resolve()
        return _read(path), {"kind": "local", "path": str(path), "sha256": _digest(path.read_bytes())}
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi

    repo = cfg.get("test_source", "HuggingFaceH4/MATH-500") if test else cfg.get("source", "EleutherAI/hendrycks_math")
    requested = cfg.get("test_revision" if test else "source_revision") or "main"
    revision = HfApi().dataset_info(repo, revision=requested).sha
    if not revision:
        raise RuntimeError(f"Cannot resolve dataset revision: {repo}@{requested}")
    names = [None] if test else cfg.get("subjects", list(SUBJECTS))
    if not names:
        names = get_dataset_config_names(repo, revision=revision)
    rows = []
    split = "test" if test else "train"
    for name in names:
        rows.extend(dict(row) for row in load_dataset(repo, name, split=split, revision=revision))
    return rows, {"kind": "huggingface", "repo": repo, "revision": revision,
                  "requested_revision": requested, "configs": names, "split": split}


def _register(name: str, rows: dict[str, list[dict]]) -> None:
    from rllm.data.dataset import DatasetRegistry

    for split in SPLITS:
        DatasetRegistry.register_dataset(name=name, data=rows[split], split=split, category="math")


def prepare(config: dict, root: Path, register: bool = True) -> dict:
    """Write data/{train,dev,test}.jsonl; reuse only an identical verified manifest.

    Remote source refs are frozen on first preparation. To intentionally refresh
    data or change the split, use a fresh project/output root. Local JSONL files
    may contain original HF fields or this module's normalized row fields.
    """
    cfg = config.get("data", {})
    root, requested = Path(root).resolve(), dict(cfg)
    for key in ("train_jsonl", "test_jsonl"):
        if cfg.get(key):
            path = (root / cfg[key]).resolve()
            requested[key] = {"path": str(path), "sha256": _digest(path.read_bytes())}
    output = root / "data"
    manifest_path = output / "manifest.json"
    name = cfg.get("registry_name", "rft_math_pilot")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("requested") != requested:
            raise ValueError("Existing data manifest differs from the requested dataset; use a fresh output root")
        rows = {}
        for split in SPLITS:
            path = output / f"{split}.jsonl"
            if not path.is_file() or _digest(path.read_bytes()) != manifest["splits"][split]["sha256"]:
                raise ValueError(f"Prepared {split} data differs from its manifest")
            rows[split] = _read(path)
        if register:
            _register(name, rows)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Non-empty data directory without a manifest; use a fresh output root")
    sizes = [cfg.get("train_size", 2048), cfg.get("dev_size", 256)]
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("train_size and dev_size must be positive integers")
    seed = cfg.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("data.seed must be an integer")
    raw_test, test_source = _source(cfg, root, test=True)
    test, removed_test = _deduplicate([_normalize(row, "test") for row in raw_test], set())
    if not test:
        raise ValueError("The held-out test dataset is empty")
    raw_train, train_source = _source(cfg, root, test=False)
    training, removed_train = _deduplicate([_normalize(row, "train") for row in raw_train],
                                          {_key(row["question"]) for row in test})
    train_size, dev_size = sizes
    if len(training) < train_size + dev_size:
        raise ValueError(f"Need {train_size + dev_size} unique non-test questions; found {len(training)}")
    training.sort(key=lambda row: _digest(f"{seed}:{_key(row['question'])}"))
    rows = {"dev": training[:dev_size], "train": training[dev_size:dev_size + train_size], "test": test}
    manifest = {"schema_version": 1, "requested": requested, "registry_name": name,
                "seed": seed, "deduplication": "NFKC and whitespace-normalized exact question hash",
                "split_method": "deterministic hash ranking; dev held out before train selection",
                "sources": {"train": train_source, "test": test_source},
                "removed": {"train_duplicates_or_test_overlap": removed_train, "test_duplicates": removed_test},
                "splits": {}}
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[split])
        path.write_bytes(payload.encode("utf-8"))
        manifest["splits"][split] = {"count": len(rows[split]), "sha256": _digest(path.read_bytes())}
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    if register:
        _register(name, rows)
    return manifest
