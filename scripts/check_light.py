"""One lightweight check: parse sources/configs and smoke-check metrics; no GPU imports."""
import ast
import contextlib
import io
import json
import sys
import tempfile
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
checked = []
for path in sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py")):
    compile(ast.parse(path.read_text(encoding="utf-8-sig")), str(path), "exec")
    checked.append(str(path.relative_to(root)))
tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

from rft_rllm.cli import plan
from rft_rllm.config import load_config
from rft_rllm.metrics import summarize_run

for path in sorted((root / "configs").glob("*.json")):
    config = load_config(path)
    with contextlib.redirect_stdout(io.StringIO()):
        plan(config, root)
    checked.append(str(path.relative_to(root)))

# Two tasks, two attempts each: one success overall -> pass@1=.25, pass@2=.5.
with tempfile.TemporaryDirectory(prefix="rft-rllm-check-") as temporary:
    directory = Path(temporary)
    result = {"attempts": 2, "total": 4, "items": [
        {"idx": i, "attempt": j, "is_correct": i == 0 and j == 0, "error": None, "signals": {}}
        for i in range(2) for j in range(2)
    ]}
    (directory / "results.json").write_text(json.dumps(result), encoding="utf-8")
    summary = summarize_run(directory, [1, 2], 2, 2)
    assert summary["pass_at"] == {"1": .25, "2": .5}
    assert summary["mean_response_tokens"] is None
    result["items"].pop()
    result["total"] -= 1
    (directory / "results.json").write_text(json.dumps(result), encoding="utf-8")
    try:
        summarize_run(directory, [1], 2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete evaluation was accepted")
print(json.dumps({"status": "passed", "files": checked, "checks": ["AST", "TOML/JSON", "three CLI plans", "tiny pass@k/incomplete-grid smoke"],
                  "not_run": ["GPU dependencies", "model/data downloads", "inference", "training"]}, ensure_ascii=False, indent=2))
