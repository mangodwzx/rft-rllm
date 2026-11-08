"""Show a no-dependency plan, or run selected official experiment stages."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import RLLM_COMMIT
from .config import load_config, project_root, set_environment
from .pipeline import Pipeline, eval_command, merger_command, select_command, sft_command
from .runtime import check_installation, server_command


def plan(config: dict, root: Path) -> None:
    output = root / "runs" / config["experiment"]["name"]
    endpoint = f"http://127.0.0.1:{config['server']['port']}/v1"
    initial = "<frozen-initial-hf-checkpoint>"
    print(f"rLLM revision: {RLLM_COMMIT}\nRun: {output}\nRLLM_HOME: {root / '.rllm'}")
    print("Prepare fixed train/dev/test data; resolve initial model revision once; snapshot packages/config/source.")
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
        print(shlex.join(sft_command(config, name, model, directory / "train")))
        print(shlex.join(merger_command(directory / "train" / "global_step_<last>", directory / "hf")))
        model = str(directory / "hf")
    print("\n# GRPO starts independently from the original frozen checkpoint")
    print(shlex.join([sys.executable, "-m", "rft_rllm.train_grpo", "--config", str(output / "config.json"),
                      "--model", initial, "--output", str(output / "grpo" / "train")]))
    for group in ("rft", "grpo"):
        print(f"\n# Evaluate {group} final HF checkpoint with the same inference settings")
        print(shlex.join(eval_command(config, "test", config["evaluation"]["attempts"], output / f"{group}_eval", endpoint)))
    print("\nReport: summary.json + summary.csv. Training budgets are recorded separately, not matched.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--config", type=Path, default=project_root() / "configs" / "baseline.json")
    parser.add_argument("--stage", choices=["prepare", "base", "rft", "grpo", "evaluate", "report", "all"], default="all")
    args = parser.parse_args(argv)
    root = project_root()
    try:
        config = load_config(args.config)
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
