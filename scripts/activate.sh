#!/usr/bin/env bash
# Every new shell must restore cache/temp paths as well as activate the venv.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/activate.sh' >&2
  exit 1
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh" || return 1
if [[ ! -x "$RFT_PROJECT_ROOT/.venv/bin/python" ]]; then
  echo 'Project environment is missing. Run: bash scripts/setup_linux.sh' >&2
  return 1
fi
source "$RFT_PROJECT_ROOT/.venv/bin/activate" || return 1
export CUDA_HOME="$("$RFT_PROJECT_ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')"
export CUDA_PATH="$CUDA_HOME"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
if [[ -x "$CUDA_HOME/bin/nvcc" && ":$PATH:" != *":$CUDA_HOME/bin:"* ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
_rft_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
case "${_rft_driver%%.*}" in
  535|570)
    _rft_compat="$RFT_PROJECT_ROOT/vendor/cuda-compat-13-0/usr/local/cuda-13.0/compat"
    if [[ ! -f "$_rft_compat/libcuda.so.1" ]]; then
      echo 'Local CUDA compatibility libraries are missing. Run: bash scripts/install_cuda_compat.sh' >&2
      unset _rft_driver _rft_compat
      return 1
    fi
    if [[ "${LD_LIBRARY_PATH:-}" != "$_rft_compat" && "${LD_LIBRARY_PATH:-}" != "$_rft_compat:"* ]]; then
      export LD_LIBRARY_PATH="$_rft_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    ;;
esac
unset _rft_driver _rft_compat
