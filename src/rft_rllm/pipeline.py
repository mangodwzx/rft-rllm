"""Glue official rLLM commands into resumable, separately budgeted baselines."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from . import RLLM_COMMIT, VERL_VERSION
from .config import digest, write_json
from .data import data_directory, load_rows, registry_name
from .metrics import compare_runs, compare_summaries, summarize_run
from .prompts import PROMPT_VERSION, prompt_fingerprint
from .runtime import Runner, finalize_raw_export, freeze_model, latest_checkpoint, require_hf_weights, serving


def eval_command(config: dict, split: str, attempts: int, output: Path, endpoint: str) -> list[str]:
    sample = config["sampling"]
    return [
        "rllm", "eval", registry_name(config, split), "--split", split,
        "--agent", "rft_rllm.flow:math_flow", "--evaluator", "rft_rllm.flow:math_evaluator",
        "--model", "rft-model", "--base-url", endpoint, "--attempts", str(attempts),
        "--concurrency", str(config["evaluation"]["concurrency"]),
        "--temperature", str(sample["temperature"]), "--top-p", str(sample["top_p"]),
        "--max-tokens", str(sample["max_tokens"]), "--episodes-dir", str(output),
        "--save-episodes", "--no-ui",
    ]


def select_command(collection: Path, name: str) -> list[str]:
    return ["rllm", "dataset", "from-eval", str(collection), "--name", name,
            "--filter", "solved", "--select", "correct", "--no-dedup", "--split", "train"]


def sft_command(config: dict, name: str, model: str, output: Path, config_path: Path | None = None) -> list[str]:
    if config_path is None:
        config_path = output.parents[2] / "config.json"
    return [
        sys.executable, "-m", "rft_rllm.train_sft", "--config", str(config_path),
        "--dataset", name, "--model", model, "--output", str(output),
    ]


def merger_command(checkpoint: Path, output: Path) -> list[str]:
    return [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
            "--local_dir", str(checkpoint), "--target_dir", str(output)]


def fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Incomplete stage already has output at {path}. Inspect it, then use a new experiment.name or archive that stage directory.")


class Pipeline:
    def __init__(self, config: dict, root: Path):
        self.config, self.root = config, root
        self.output = root / "runs" / config["experiment"]["name"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "state.json"
        source_hash = hashlib.sha256()
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.suffix not in {".py", ".json"}:
                continue
            source_hash.update(path.name.encode() + path.read_bytes())
        self.identity = {"config_sha256": digest(config), "source_sha256": source_hash.hexdigest(), "rllm_commit": RLLM_COMMIT}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["identity"] != self.identity:
                raise ValueError("Configuration or source changed since this run. Choose a new experiment.name.")
        else:
            self.state = {"identity": self.identity, "completed": {}, "models": {}}
            write_json(self.output / "config.json", config)
            write_json(self.output / "environment.json", {
                "python": sys.version, "rllm_commit": RLLM_COMMIT, "verl": VERL_VERSION,
                "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
            })
            self.save()
        self.runner = Runner(root, self.output)

    def save(self) -> None:
        write_json(self.state_path, self.state)

    def done(self, name: str, **details) -> None:
        self.state["completed"][name] = details
        self.save()

    def prepare(self) -> None:
        from .data import prepare

        manifest = prepare(self.config, self.root)
        previous = self.state.get("data_sha256")
        fingerprint = digest(manifest)
        if previous and previous != fingerprint:
            raise ValueError("Prepared data no longer matches this run")
        self.state["data_sha256"] = fingerprint
        self.counts = {name: data["count"] for name, data in manifest["splits"].items()}
        self.done("prepare", counts=self.counts)

    def initial_model(self) -> str:
        model = freeze_model(self.config, self.output / "initial_model.json", self.root)
        self.state["models"]["base"] = model
        self.save()
        return model

    def evaluate_model(self, model: str, name: str, split: str, attempts: int, output: Path) -> dict:
        ks = [1, attempts] if split == "train" else self.config["evaluation"]["ks"]
        rows = load_rows(self.config, self.root, split)
        if name in self.state["completed"]:
            return summarize_run(output, ks, attempts, self.counts[split], task_rows=rows)
        fresh_directory(output)
        with serving(self.runner, self.config, model, f"{name}_server") as endpoint:
            self.runner.run(eval_command(self.config, split, attempts, output, endpoint), name)
        summary = summarize_run(output, ks, attempts, self.counts[split], task_rows=rows)
        write_json(output / "summary.json", summary)
        write_json(output / "protocol.json", {
            "identity": self.identity, "data_sha256": self.state["data_sha256"], "model_path": model,
            "split": split, "registry_name": registry_name(self.config, split),
            "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(),
            "task_ids": [row["id"] for row in rows],
        })
        self.done(name, output=str(output))
        return summary

    def base(self) -> None:
        self.evaluate_model(self.initial_model(), "base_eval", "test", self.config["evaluation"]["attempts"], self.output / "base_eval")

    def debug(self) -> None:
        if self.config["data"].get("kind") != "gsm8k":
            raise ValueError("Use configs/gsm8k_debug.json for the generation/grading debug stage")
        self.evaluate_model(self.initial_model(), "debug_eval", "debug", self.config["evaluation"]["attempts"], self.output / "debug_eval")

    def rft(self) -> None:
        from rllm.data import DatasetRegistry

        model = self.initial_model()
        for index in range(1, self.config["rft"]["rounds"] + 1):
            name = f"rft_round_{index:02d}"
            directory = self.output / "rft" / f"round_{index:02d}"
            collection = directory / "collect"
            stats = self.evaluate_model(model, name + "_collect", "train", self.config["rft"]["attempts"], collection)
            if stats["correct"] == 0:
                raise RuntimeError("RFT accepted zero responses. Inspect collection results before changing model or sampling settings.")
            curated_name = f"{self.config['experiment']['name']}_{name}_correct"
            if name + "_select" not in self.state["completed"]:
                self.runner.run(select_command(collection, curated_name), name + "_select")
                self.done(name + "_select", dataset=curated_name)
            curated = DatasetRegistry.load_dataset(curated_name, "train")
            if curated is None or len(curated) != stats["correct"]:
                raise RuntimeError("Curated row count differs from correct rollout count; inspect official curation logs.")
            if len(curated) < self.config["rft"]["sft"]["batch_size"]:
                raise ValueError("Too few accepted rows for SFT batch_size. Use a smaller batch_size in a new experiment.")
            training = directory / "train"
            if name + "_sft" not in self.state["completed"]:
                fresh_directory(training)
                self.runner.run(sft_command(self.config, curated_name, model, training, self.output / "config.json"), name + "_sft")
                checkpoint = latest_checkpoint(training, actor=False)
                self.done(name + "_sft", checkpoint=str(checkpoint), input_model=model, accepted_rows=len(curated),
                          generated_responses=stats["total"], observed_generated_tokens=stats["observed_response_tokens"],
                          epochs=self.config["rft"]["sft"]["epochs"], trainer_steps=int(checkpoint.name[12:]))
            checkpoint = Path(self.state["completed"][name + "_sft"]["checkpoint"])
            exported = directory / "hf"
            if name + "_export" not in self.state["completed"]:
                fresh_directory(exported)
                self.runner.run(merger_command(checkpoint, exported), name + "_export")
                finalize_raw_export(exported)
                self.done(name + "_export", path=str(exported))
            require_hf_weights(exported)
            model = str(exported)
        self.state["models"]["rft"] = model
        self.save()

    def grpo(self) -> None:
        model = self.initial_model()
        training = self.output / "grpo" / "train"
        if "grpo_train" not in self.state["completed"]:
            fresh_directory(training)
            self.runner.run([sys.executable, "-m", "rft_rllm.train_grpo", "--config", str(self.output / "config.json"),
                             "--model", model, "--output", str(training)], "grpo_train")
            checkpoint = latest_checkpoint(training, actor=True)
            self.done("grpo_train", checkpoint=str(checkpoint), input_model=model, trainer_steps=int(checkpoint.parent.name[12:]))
        checkpoint = Path(self.state["completed"]["grpo_train"]["checkpoint"])
        exported = checkpoint / "huggingface"
        if not list(exported.glob("*.safetensors")) and not list(exported.glob("pytorch_model*.bin")):
            exported = self.output / "grpo" / "hf"
            if "grpo_export" not in self.state["completed"]:
                fresh_directory(exported)
                self.runner.run(merger_command(checkpoint, exported), "grpo_export")
                require_hf_weights(exported)
                self.done("grpo_export", path=str(exported))
        require_hf_weights(exported)
        finalize_raw_export(exported)
        self.state["models"]["grpo"] = str(exported)
        self.save()

    def evaluate(self) -> None:
        for name in ("base", "rft", "grpo"):
            model = self.state["models"].get(name)
            if not model:
                raise RuntimeError(f"Run stage {name} before evaluate")
            self.evaluate_model(model, name + "_eval", "test", self.config["evaluation"]["attempts"], self.output / f"{name}_eval")

    def report(self) -> None:
        from .budget import collect_budgets
        from .data import prepare

        # Validate the frozen files even for report-only runs (no GPU imports).
        if not (data_directory(self.config, self.root) / "manifest.json").is_file():
            raise FileNotFoundError("Prepared data manifest is missing; report never downloads or reconstructs data")
        manifest = prepare(self.config, self.root, register=False)
        if digest(manifest) != self.state.get("data_sha256"):
            raise ValueError("Prepared data no longer matches the evaluated run")
        rows = load_rows(self.config, self.root, "test")
        count = self.state["completed"]["prepare"]["counts"]["test"]
        paths = {name: self.output / f"{name}_eval" for name in ("base", "rft", "grpo")}
        for name, path in paths.items():
            if name + "_eval" not in self.state["completed"]:
                raise RuntimeError(f"Missing completed {name} evaluation")
            protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
            if protocol["identity"] != self.identity or protocol["data_sha256"] != self.state["data_sha256"] or protocol["split"] != "test":
                raise ValueError(f"Evaluation protocol mismatch: {path}")
            if protocol.get("task_ids") != [row["id"] for row in rows] or protocol.get("prompt_sha256") != prompt_fingerprint():
                raise ValueError(f"Evaluation task identity or prompt mismatch: {path}")
        summaries = compare_runs(paths, self.config["evaluation"]["ks"], self.config["evaluation"]["attempts"], count, task_rows=rows)
        for name, summary in summaries.items():
            summary["checkpoint"] = self.state["models"][name]
        budget = collect_budgets(self.output, self.state, self.config)
        write_json(self.output / "budget.json", budget)
        comparisons = compare_summaries(summaries)
        write_json(self.output / "comparison.json", comparisons)
        from .reporting import write_comparison_artifacts

        write_comparison_artifacts(self.output, summaries, comparisons)
        write_json(self.output / "summary.json", {"groups": summaries, "comparisons": comparisons, "training_budget": budget,
                   "note": "Offline/iterative RFT versus online GRPO; generation and optimizer budgets are not equal."})
        columns = ["group", "subset", "checkpoint", "tasks", "attempts", "accuracy", "mean_response_tokens", "mean_correct_response_tokens", "truncation_rate", "errors"]
        columns += [f"pass@{k}" for k in self.config["evaluation"]["ks"]]
        with (self.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for name, info in summaries.items():
                subsets = {"all": info, "level_4_5": info["subsets"]["level_4_5"]}
                subsets.update({f"level_{key}": value for key, value in info["subsets"]["by_level"].items()})
                subsets.update({f"subject_{key}": value for key, value in info["subsets"]["by_subject"].items()})
                for subset, metrics in subsets.items():
                    row = {key: metrics[key] for key in columns if key in metrics}
                    row.update({"group": name, "subset": subset, "checkpoint": info["checkpoint"],
                                **{f"pass@{k}": value for k, value in (metrics["pass_at"] or {}).items()}})
                    writer.writerow(row)
        print(f"Saved {self.output / 'summary.csv'}")

    def run(self, stage: str) -> None:
        debug_only = self.config["data"].get("kind") == "gsm8k"
        if debug_only and stage not in {"all", "prepare", "debug"}:
            raise ValueError("GSM8K is for generation/grading only; use stage debug or prepare")
        if stage == "report":
            self.report()
            return
        self.prepare()
        if stage == "prepare":
            return
        if debug_only:
            self.debug()
            return
        if stage == "all":
            self.base()
            self.rft()
            self.grpo()
            self.evaluate()
            self.report()
        else:
            getattr(self, stage)()
