"""Process orchestration and owned local inference server."""
from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import RLLM_COMMIT, VERL_VERSION
from .config import write_json


def check_installation() -> None:
    dist = importlib.metadata.distribution("rllm")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    if not revision and direct.get("dir_info", {}).get("editable"):
        from urllib.parse import unquote, urlparse

        source = Path(unquote(urlparse(direct["url"]).path))
        if os.name == "nt" and str(source).startswith("\\"):
            source = Path(str(source).lstrip("\\"))
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty.strip():
            raise RuntimeError("The pinned rLLM checkout has tracked modifications; use a clean checkout.")
    if revision != RLLM_COMMIT:
        raise RuntimeError(f"Expected rLLM {RLLM_COMMIT}, found {revision!r}. Run scripts/setup_linux.sh.")
    if importlib.metadata.version("verl") != VERL_VERSION:
        raise RuntimeError(f"Expected verl=={VERL_VERSION}")


class Runner:
    def __init__(self, root: Path, run_dir: Path):
        self.root, self.run_dir = root, run_dir

    def record(self, argv: list[str], label: str, **extra) -> None:
        item = {"time": datetime.now(timezone.utc).isoformat(), "label": label, "argv": argv, **extra}
        with (self.run_dir / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def run(self, argv: list[str], label: str) -> None:
        print(f"[{label}] {shlex.join(argv)}", flush=True)
        self.record(argv, label)
        log = self.run_dir / "logs" / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(argv, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        self.record(argv, label, returncode=result.returncode, elapsed_seconds=time.monotonic() - start, log=str(log))
        if result.returncode:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); inspect {log}")


def server_command(config: dict, model: str) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--served-model-name", "rft-model", "--host", config["server"]["host"],
        "--port", str(config["server"]["port"]), "--seed", str(config["server"]["seed"]),
        "--tensor-parallel-size", str(config["resources"]["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(config["resources"]["gpu_memory_utilization"]),
        "--max-model-len", str(config["sampling"]["max_tokens"] + config["sampling"]["max_prompt_length"]),
        "--generation-config", "vllm",
    ]


@contextmanager
def serving(runner: Runner, config: dict, model: str, label: str):
    host, port = config["server"]["host"], config["server"]["port"]
    with socket.socket() as probe:
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError(f"Port {port} is occupied; choose another server.port. Existing servers are not reused.")
    argv = server_command(config, model)
    runner.record(argv, label)
    log = runner.run_dir / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://{host}:{port}/v1"
    # Local requests must not inherit an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=runner.root, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + config["server"]["startup_timeout"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited before readiness; inspect {log}")
                try:
                    with opener.open(endpoint + "/models", timeout=2) as response:
                        models = json.load(response)
                    if any(item["id"] == "rft-model" for item in models["data"]):
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"vLLM startup timed out; inspect {log}")
                time.sleep(2)
            yield endpoint
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


def freeze_model(config: dict, destination: Path, root: Path) -> str:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    if destination.exists():
        info = json.loads(destination.read_text(encoding="utf-8"))
        model_path = Path(info["path"])
        if not model_path.is_dir():
            raise FileNotFoundError(f"Frozen initial model no longer exists: {model_path}")
        return str(model_path)
    spec = config["model"]
    if spec.get("local_path"):
        model_path = (root / spec["local_path"]).resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        info = {"source": "local", "path": str(model_path), "note": "Keep this local checkpoint immutable."}
    else:
        revision = HfApi().model_info(spec["repo_id"], revision=spec.get("revision") or "main").sha
        model_path = Path(snapshot_download(spec["repo_id"], revision=revision)).resolve()
        info = {"source": "huggingface", "repo_id": spec["repo_id"], "revision": revision, "path": str(model_path)}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if not tokenizer.chat_template:
        raise ValueError("The official chat workflow requires a tokenizer chat_template. Choose a chat model or supply a prepared common checkpoint.")
    write_json(destination, info)
    return str(model_path)


def latest_checkpoint(path: Path, actor: bool) -> Path:
    candidates = [entry for entry in path.glob("global_step_*") if entry.is_dir() and entry.name[12:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {path}; inspect the training log")
    result = max(candidates, key=lambda entry: int(entry.name[12:]))
    return result / "actor" if actor else result


def require_hf_weights(path: Path) -> None:
    if not (path / "config.json").is_file() or not (
        list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))
    ):
        raise FileNotFoundError(f"No exported Hugging Face weights in {path}")
