"""Show a no-dependency plan, or run selected official experiment stages."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import RLLM_COMMIT
from .config import MODELS, load_config, project_root, select_model, set_environment, validate
from .data import data_directory
from .pipeline import Pipeline, eval_command, merger_command, select_command, sft_command
from .runtime import check_installation, server_command


def plan(config: dict, root: Path) -> None:
    output = root / "runs" / config["experiment"]["name"]
    endpoint = f"http://127.0.0.1:{config['server']['port']}/v1"
    initial = "<frozen-initial-hf-checkpoint>"
    print(f"rLLM revision: {RLLM_COMMIT}\nRun: {output}\nRLLM_HOME: {root / '.rllm'}")
    print(f"Model: {config['model'].get('local_path') or config['model']['repo_id']} ({config['model'].get('revision', 'main')})")
    print(f"Data: {data_directory(config, root)}; kind={config['data'].get('kind', 'math')}")
    print(f"Evaluation: N={config['evaluation']['attempts']} per task, k={config['evaluation']['ks']}")
    print("Prompt: fixed raw-completion few-shot examples; Base checkpoints are prepared without changing source weights.")
    if config["data"].get("kind") == "gsm8k":
        print(f"\n# GSM8K debug: {config['data'].get('debug_size', 32)} tasks; generation and grading only")
        print(shlex.join(server_command(config, initial)))
        print(shlex.join(eval_command(config, "debug", config["evaluation"]["attempts"], output / "debug_eval", endpoint)))
        return
    print(f"Training: {config['data'].get('train_size', 2048)} tasks; fixed dev held out before pilot/full selection.")
    print("\n# Base evaluation (each evaluation stage owns/stops its vLLM server)")
    print(shlex.join(server_command(config, initial)))
    print(shlex.join(eval_command(config, "test", config["evaluation"]["attempts"], output / "base_eval", endpoint)))
    model = initial
    for index in range(1, config["rft"]["rounds"] + 1):
        directory = output / "rft" / f"round_{index:02d}"
        name = f"{config['experiment']['name']}_rft_round_{index:02d}_correct"
        print(f"\n# RFT round {index}: collect from {model}")
        print(shlex.join(eval_command(config, "train", config["rft"]["attempts"], directory / "collect", endpoint)))
        print(shlex.join(select_command(directory / "collect", name)))
        print(shlex.join(sft_command(config, name, model, directory / "train", output / "config.json")))
        print(shlex.join(merger_command(directory / "train" / "global_step_<last>", directory / "hf")))
        model = str(directory / "hf")
    print("\n# GRPO starts independently from the original frozen checkpoint")
    print(shlex.join([sys.executable, "-m", "rft_rllm.train_grpo", "--config", str(output / "config.json"),
                      "--model", initial, "--output", str(output / "grpo" / "train")]))
    for group in ("rft", "grpo"):
        print(f"\n# Evaluate {group} final HF checkpoint with the same inference settings")
        print(shlex.join(eval_command(config, "test", config["evaluation"]["attempts"], output / f"{group}_eval", endpoint)))
    print("\nReport: overall / level 4–5 / level / subject pass@k, GRPO−RFT deltas and curves.")
    print("RFT uses official offline SFT on accepted responses; training budgets are recorded separately, not matched.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "models"])
    parser.add_argument("--config", type=Path, default=project_root() / "configs" / "baseline.json")
    parser.add_argument("--stage", choices=["prepare", "debug", "base", "rft", "grpo", "evaluate", "report", "all"], default="all")
    parser.add_argument("--model", choices=list(MODELS), help="Select a Base model; also gives it a distinct run name")
    parser.add_argument("--name", help="Explicit experiment.name for this run")
    parser.add_argument("--revision", help="HF revision; a 40-character commit can be resolved from local cache offline")
    parser.add_argument("--local-model", type=Path, help="Existing immutable HF Base checkpoint")
    parser.add_argument("--gpus", type=int, help="GPUs used by training (choose physical devices with CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--tensor-parallel-size", type=int, help="Inference tensor parallelism")
    args = parser.parse_args(argv)
    root = project_root()
    try:
        if args.command == "models":
            for key, repo_id in MODELS.items():
                print(f"{key}\t{repo_id}")
            return
        config = load_config(args.config)
        if args.model:
            config = select_model(config, args.model)
        if args.name:
            config["experiment"]["name"] = args.name
        if args.revision:
            config["model"]["revision"] = args.revision
        if args.local_model:
            config["model"]["local_path"] = str(args.local_model.resolve())
        if args.gpus is not None:
            config["resources"]["gpus"] = args.gpus
        if args.tensor_parallel_size is not None:
            config["resources"]["tensor_parallel_size"] = args.tensor_parallel_size
        validate(config)
        if args.command == "plan":
            plan(config, root)
            return
        set_environment(root)
        if args.stage != "report":
            check_installation()
        Pipeline(config, root).run(args.stage)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
