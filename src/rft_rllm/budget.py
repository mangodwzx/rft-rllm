"""Observed budgets and separately labelled plans from saved run artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _telemetry(records: list[dict]) -> dict:
    """Do not turn an absent usage field into a zero-token observation."""
    output = {}
    coverage = {}
    for name in ("response_tokens", "prompt_tokens", "truncated"):
        values = [_number(record.get(name)) for record in records]
        known = [value for value in values if value is not None]
        coverage[name] = len(known)
        if name == "truncated":
            output["observed_truncation_rate"] = sum(known) / len(known) if known else None
        else:
            output[f"observed_{name}"] = sum(known) if known else None
    output["telemetry_coverage"] = {**coverage, "records": len(records)}
    return output


def _durations(run_dir: Path) -> dict:
    phases: dict[str, dict] = defaultdict(lambda: {"elapsed_seconds": 0.0, "completed_commands": 0, "failed_commands": 0})
    path = run_dir / "commands.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            elapsed = _number(record.get("elapsed_seconds"))
            if "returncode" not in record or elapsed is None:
                continue
            phase = phases[str(record.get("label", "unknown"))]
            phase["elapsed_seconds"] += elapsed
            phase["completed_commands"] += 1
            phase["failed_commands"] += int(record["returncode"] != 0)
    return {
        "phases": dict(phases),
        # An owned server overlaps its eval command; avoid double-counting it.
        "total_recorded_command_seconds": sum(
            value["elapsed_seconds"] for name, value in phases.items() if not name.endswith("_server")
        ) if phases else None,
        "scope": "Completed logged commands, including failed attempts; excludes model/data downloads, unrecorded work and overlapping server lifetimes. Not GPU-hours.",
    }


def _rft_rounds(run_dir: Path, state: dict) -> list[dict]:
    rounds = []
    for name, info in sorted(state.get("completed", {}).items()):
        if not name.startswith("rft_round_") or not name.endswith("_sft"):
            continue
        number = int(name.removeprefix("rft_round_").removesuffix("_sft"))
        collection = run_dir / "rft" / f"round_{number:02d}" / "collect"
        summary = _read(collection / "summary.json")
        result = _read(collection / "results.json")
        correct_items = [item for item in result.get("items", []) if item.get("is_correct") and item.get("error") is None]
        accepted_telemetry = _telemetry([item.get("signals") or {} for item in correct_items])
        row = {key: value for key, value in info.items() if key != "updates"}
        row.update({
            "round": number,
            "trainer_steps": info.get("trainer_steps", info.get("updates")),
            "collection_acceptance_rate": summary.get("acceptance_rate"),
            "collection_observed_task_coverage": summary.get("observed_task_coverage"),
            "generated_responses": summary.get("total", info.get("generated_responses")),
            "observed_generated_tokens": summary.get("observed_response_tokens", info.get("observed_generated_tokens")),
            "collection_token_coverage": summary.get("coverage_counts", {}),
            "accepted_response_tokens_observed": accepted_telemetry["observed_response_tokens"],
            "accepted_response_token_coverage": accepted_telemetry["telemetry_coverage"]["response_tokens"],
            "correct_collection_responses": len(correct_items) if result else None,
            "actual_backprop_tokens": None,
            "backprop_tokens_reason": "Accepted response usage precedes SFT retokenization, masking, dropped batches and epochs; it is not an observed backward-pass token count. Inspect the official SFT log.",
            "log": str(run_dir / "logs" / f"{name}.log"),
        })
        rounds.append(row)
    return rounds


def _grpo(run_dir: Path, state: dict, config: dict) -> dict:
    training = run_dir / "grpo" / "train"
    plan = _read(training / "budget_plan.json")
    requested = config["grpo"]
    train_tasks = plan.get("train_tasks", state.get("completed", {}).get("prepare", {}).get("counts", {}).get("train"))
    # Effective values belong in budget_plan.json. Requested config is usable
    # for planning only when arbitrary Hydra overrides cannot supersede it.
    unambiguous = not requested.get("overrides")
    batch = plan.get("train_batch_size", requested["batch_size"] if unambiguous else None)
    group = plan.get("group_size", requested["group_size"] if unambiguous else None)
    epochs = plan.get("epochs", requested["epochs"] if unambiguous else None)
    steps = plan.get("planned_trainer_steps", plan.get("planned_updates"))
    if steps is None and train_tasks is not None and batch and epochs:
        steps = (train_tasks // batch) * epochs
        if requested.get("max_steps"):
            steps = min(steps, requested["max_steps"])
    task_exposures = steps * batch if steps is not None and batch is not None else None
    planned_responses = task_exposures * group if task_exposures is not None and group is not None else None

    records = []
    logged_steps = set()
    rollout_root = training / "rollouts"
    for path in sorted(rollout_root.rglob("episode_*.json")):
        step_directories = [part for part in path.relative_to(rollout_root).parts[:-1] if part.startswith("train_step")]
        if not step_directories:
            continue
        episode = _read(path)
        records.append(episode.get("metrics") or {})
        logged_steps.add(str(path.parent))
    observed = _telemetry(records)
    info = state.get("completed", {}).get("grpo_train", {})
    return {
        **{key: value for key, value in info.items() if key != "updates"},
        "trainer_steps": info.get("trainer_steps", info.get("updates")),
        "optimizer_updates": None,
        "optimizer_updates_reason": "A saved trainer step counts a rollout batch; actor minibatches, PPO epochs or skipped groups can change optimizer steps. No optimizer counter was collected.",
        "plan": {
            "train_tasks": train_tasks,
            "train_batch_size": batch,
            "group_size": group,
            "epochs": epochs,
            "trainer_steps": steps,
            "task_exposures": task_exposures,
            "responses_without_retries": planned_responses,
            "source": str(training / "budget_plan.json") if plan else "Requested config; no effective trainer plan saved",
        },
        "logged_training_episodes": len(records),
        "logged_training_step_directories": len(logged_steps),
        **observed,
        "actual_generated_responses": None,
        "actual_generated_responses_reason": "Logged training episodes may omit retries or other unlogged generation. Logged episodes are observations, not proof that all were used in a policy update.",
        "actual_backprop_tokens": None,
        "backprop_tokens_reason": "Rollout token usage does not account for training masks, minibatches, PPO reuse or discarded trajectories. No backward-pass token counter was collected.",
        "rollout_log_dir": str(rollout_root),
        "log": str(run_dir / "logs" / "grpo_train.log"),
    }


def collect_budgets(run_dir: Path, state: dict, config: dict) -> dict:
    """Collect observations; keep requested/planned quantities explicitly separate."""
    run_dir = Path(run_dir)
    return {
        "rft": _rft_rounds(run_dir, state),
        "grpo": _grpo(run_dir, state, config),
        "timing": _durations(run_dir),
        "note": "Shared task/evaluation protocol does not match generated-token, backprop-token or optimizer budgets. Null counters were not measured.",
    }
