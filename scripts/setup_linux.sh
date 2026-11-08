#!/usr/bin/env bash
# Install only. No model downloads, inference, or training are started here.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="4e828c81680e5a85b638ceedeaf09d1e90109ca8"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this installer on the Linux/CUDA training machine.' >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
command -v git >/dev/null
mkdir -p "$ROOT/vendor"
if [[ ! -d "$ROOT/vendor/rllm" ]]; then
  git clone https://github.com/rllm-org/rllm.git "$ROOT/vendor/rllm"
fi
if [[ -n "$(git -C "$ROOT/vendor/rllm" status --porcelain --untracked-files=no)" ]]; then
  echo 'vendor/rllm has tracked changes; preserve them before installing.' >&2
  exit 1
fi
git -C "$ROOT/vendor/rllm" fetch origin "$REVISION"
git -C "$ROOT/vendor/rllm" checkout --detach "$REVISION"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.11 "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"
cd "$ROOT/vendor/rllm"
# Follow the pinned project's uv sources/build-dependency declarations.
# Override TORCH_BACKEND if the target GPU environment requires another build.
uv pip install --python "$ROOT/.venv/bin/python" -e '.[verl,rewards]' -e ./rllm-model-gateway --torch-backend="${TORCH_BACKEND:-cu128}"
uv pip install --python "$ROOT/.venv/bin/python" --no-deps -e "$ROOT"
mkdir -p "$ROOT/environment"
uv pip freeze --python "$ROOT/.venv/bin/python" > "$ROOT/environment/requirements-resolved.txt"
git rev-parse HEAD > "$ROOT/environment/rllm-revision.txt"
printf 'Installed. Activate with: source %q/.venv/bin/activate\n' "$ROOT"
