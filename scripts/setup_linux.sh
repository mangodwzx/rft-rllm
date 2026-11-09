#!/usr/bin/env bash
# Install only. No model downloads, inference, or training are started here.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
REVISION="4e828c81680e5a85b638ceedeaf09d1e90109ca8"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this installer on the Linux/CUDA training machine.' >&2
  exit 1
fi
command -v uv >/dev/null || {
  echo 'Install uv inside this repository first; see the installation commands in README.md.' >&2
  exit 1
}
command -v git >/dev/null
command -v g++ >/dev/null
if [[ "${TORCH_BACKEND:-cu130}" != cu130 ]]; then
  echo 'This installer matches the pinned vLLM 0.22.1 default CUDA 13 wheel; use TORCH_BACKEND=cu130.' >&2
  exit 1
fi
mkdir -p "$ROOT/environment"
exec > >(tee -a "$ROOT/environment/install.log") 2>&1
# A800 on the 535/570 driver families can use NVIDIA's local forward-compat
# libraries. This only extracts files into vendor; no sudo or driver changes.
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
case "${DRIVER_VERSION%%.*}" in
  535|570)
    bash "$ROOT/scripts/install_cuda_compat.sh"
    COMPAT_DIR="$ROOT/vendor/cuda-compat-13-0/usr/local/cuda-13.0/compat"
    export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
esac
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
  uv python install 3.11
  uv venv --python 3.11 --managed-python "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"
cd "$ROOT/vendor/rllm"
# vLLM 0.22.1 requires torch 2.11.0. flash-attn 2.8.3 has no corresponding
# prebuilt wheel, so install a matching CUDA compiler inside the venv first.
# Explicit component extras prevent nvcc from pulling newer, mismatched CRTs.
uv pip install --python "$ROOT/.venv/bin/python" --torch-backend=cu130 \
  'torch==2.11.0' 'cuda-toolkit[nvcc,cccl,crt,cudart,nvrtc,nvvm]==13.0.2' \
  packaging setuptools wheel ninja
bash "$ROOT/scripts/prepare_cuda.sh"
export CUDA_HOME="$("$ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
# Follow the pinned project's dependency overrides/build dependencies.
uv pip install --python "$ROOT/.venv/bin/python" \
  --constraints "$ROOT/constraints/runtime.txt" \
  -e '.[verl,rewards]' -e ./rllm-model-gateway --torch-backend=cu130
uv pip install --python "$ROOT/.venv/bin/python" --no-deps -e "$ROOT"
uv pip freeze --python "$ROOT/.venv/bin/python" > "$ROOT/environment/requirements-resolved.txt"
git rev-parse HEAD > "$ROOT/environment/rllm-revision.txt"
uv --version > "$ROOT/environment/uv-version.txt"
printf 'cu130\n' > "$ROOT/environment/torch-backend.txt"
printf 'Installed. In each new shell run: source %q/scripts/activate.sh\n' "$ROOT"
