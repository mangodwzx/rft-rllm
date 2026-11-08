"""Summarize complete official rLLM eval runs without GPU dependencies."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _signal(item: dict, name: str) -> float | None:
    value = (item.get("signals") or {}).get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        value = float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name} signal at idx={item['idx']}, attempt={item['attempt']}")
    if name == "truncated" and value not in (0, 1):
        raise ValueError("truncated signals must be 0 or 1")
    return float(value)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def summarize_run(
    run_dir: Path,
    ks: list[int],
    expected_attempts: int | None = None,
    expected_tasks: int | None = None,
) -> dict:
    """Read ``results.json``; refuse incomplete or duplicated task/attempt grids.

    Error attempts remain in every denominator and count as failures. Missing
    telemetry is unknown (``None``), never an inferred zero. ``coverage_counts``
    reports the denominator actually available for each telemetry mean.
    """
    result_path = Path(run_dir) / "results.json"
    with result_path.open(encoding="utf-8-sig") as handle:
        result = json.load(handle)
    attempts = _integer(result.get("attempts", 1), "attempts", minimum=1)
    if expected_attempts is not None:
        _integer(expected_attempts, "expected_attempts", minimum=1)
        if attempts != expected_attempts:
            raise ValueError(f"Expected {expected_attempts} attempts, found {attempts}: {result_path}")
    if not ks:
        raise ValueError("Provide at least one pass@k value")
    ks = sorted({_integer(k, "k", minimum=1) for k in ks})
    if ks[-1] > attempts:
        raise ValueError(f"pass@{ks[-1]} requires at least {ks[-1]} attempts; run has {attempts}")

    items = result.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"No rollout items in {result_path}")
    if result.get("total") != len(items):
        raise ValueError(f"Reported total disagrees with item count in {result_path}")
    by_task: dict[int, dict[int, dict]] = defaultdict(dict)
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every rollout item must be an object")
        idx = _integer(item.get("idx"), "idx")
        attempt = _integer(item.get("attempt", 0), "attempt")
        if not isinstance(item.get("is_correct"), bool):
            raise ValueError(f"is_correct must be a boolean for task {idx}, attempt {attempt}")
        if attempt >= attempts:
            raise ValueError(f"Out-of-range attempt {attempt} for task {idx}")
        if attempt in by_task[idx]:
            raise ValueError(f"Duplicate rollout for task {idx}, attempt {attempt}")
        if item.get("signals") is not None and not isinstance(item["signals"], dict):
            raise ValueError(f"signals must be an object for task {idx}, attempt {attempt}")
        # Legacy one-attempt results may omit the attempt field.
        item = dict(item, attempt=attempt)
        by_task[idx][attempt] = item

    task_count = len(by_task) if expected_tasks is None else _integer(expected_tasks, "expected_tasks", minimum=1)
    if set(by_task) != set(range(task_count)):
        raise ValueError(f"Expected task indices 0..{task_count - 1}; got {sorted(by_task)}")
    for idx, group in by_task.items():
        missing = set(range(attempts)) - set(group)
        if missing:
            raise ValueError(f"Incomplete task {idx}: missing attempts {sorted(missing)}")

    counts: list[int] = []
    response_tokens: list[float] = []
    correct_response_tokens: list[float] = []
    prompt_tokens: list[float] = []
    response_chars: list[float] = []
    truncated: list[float] = []
    errors = 0
    for group in by_task.values():
        correct = 0
        for item in group.values():
            failed = item.get("error") is not None
            errors += int(failed)
            successful = item["is_correct"] and not failed
            correct += int(successful)
            length = _signal(item, "response_tokens")
            if length is not None:
                response_tokens.append(length)
                if successful:
                    correct_response_tokens.append(length)
            for name, target in (
                ("prompt_tokens", prompt_tokens),
                ("response_chars", response_chars),
                ("truncated", truncated),
            ):
                value = _signal(item, name)
                if value is not None:
                    target.append(value)
        counts.append(correct)

    total = task_count * attempts
    correct = sum(counts)
    pass_at = {
        str(k): fmean(
            1.0 if attempts - count < k else 1.0 - math.comb(attempts - count, k) / math.comb(attempts, k)
            for count in counts
        )
        for k in ks
    }
    return {
        "run_dir": str(Path(run_dir).resolve()),
        "dataset_name": result.get("dataset_name"),
        "model": result.get("model"),
        "agent": result.get("agent"),
        "tasks": task_count,
        "attempts": attempts,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total,
        # For solved/correct selection without caps or dedup, this is the
        # correctness acceptance rate before any missing-message exclusions.
        "acceptance_rate": correct / total,
        "observed_task_coverage": sum(count > 0 for count in counts) / task_count,
        "pass_at": pass_at,
        "mean_response_tokens": _mean(response_tokens),
        "mean_correct_response_tokens": _mean(correct_response_tokens),
        "mean_prompt_tokens": _mean(prompt_tokens),
        "mean_response_chars": _mean(response_chars),
        "truncation_rate": _mean(truncated),
        "observed_response_tokens": sum(response_tokens) if response_tokens else None,
        "observed_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "coverage_counts": {
            "response_tokens": len(response_tokens),
            "correct_response_tokens": len(correct_response_tokens),
            "prompt_tokens": len(prompt_tokens),
            "response_chars": len(response_chars),
            "truncated": len(truncated),
            "all_rollouts": total,
            "correct_rollouts": correct,
        },
    }


def compare_runs(
    named_runs: dict[str, Path],
    ks: list[int],
    expected_attempts: int | None = None,
    expected_tasks: int | None = None,
) -> dict[str, dict]:
    """Summarize named arms using one requested evaluation protocol.

    The experiment manifest must additionally establish identical task content,
    order, prompt and decoding settings; dataset names alone cannot prove this.
    """
    if not named_runs:
        raise ValueError("Provide at least one evaluation run")
    summaries = {
        name: summarize_run(path, ks, expected_attempts, expected_tasks)
        for name, path in named_runs.items()
    }
    protocols = {(row["dataset_name"], row["tasks"], row["attempts"], row["agent"]) for row in summaries.values()}
    if len(protocols) != 1:
        raise ValueError("Compared runs must use the same dataset, task count, attempts and agent")
    return summaries
