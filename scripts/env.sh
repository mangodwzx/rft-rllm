#!/usr/bin/env bash
# Source before installing anything, and before every experiment session.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/env.sh' >&2
  exit 1
fi

export RFT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RFT_RUNTIME_DIR="$RFT_PROJECT_ROOT/.runtime"
export XDG_CACHE_HOME="$RFT_RUNTIME_DIR/cache"
export XDG_CONFIG_HOME="$RFT_RUNTIME_DIR/config"
export XDG_DATA_HOME="$RFT_RUNTIME_DIR/share"
export XDG_STATE_HOME="$RFT_RUNTIME_DIR/state"

export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$RFT_RUNTIME_DIR/python"
export UV_PYTHON_BIN_DIR="$RFT_RUNTIME_DIR/bin"
export UV_TOOL_DIR="$RFT_RUNTIME_DIR/uv-tools"
export UV_TOOL_BIN_DIR="$RFT_RUNTIME_DIR/bin"
export UV_PROJECT_ENVIRONMENT="$RFT_PROJECT_ROOT/.venv"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PYTHONUSERBASE="$RFT_RUNTIME_DIR/python-user"
export PYTHONNOUSERSITE=1

export HF_HOME="$RFT_RUNTIME_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
# Remove a legacy override inherited from another experiment environment.
unset TRANSFORMERS_CACHE
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TRITON_HOME="$XDG_CACHE_HOME/triton"
export CCACHE_DIR="$XDG_CACHE_HOME/ccache"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"

export TMPDIR="$RFT_RUNTIME_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export VLLM_RPC_BASE_PATH="$TMPDIR"
# Ray appends /ray/session_.../sockets/...; a deeper base exceeds Linux's
# Unix-socket pathname limit in this repository. Files still stay in the repo.
export RAY_TMPDIR="$RFT_PROJECT_ROOT"
export RLLM_HOME="$RFT_PROJECT_ROOT/.rllm"
export WANDB_DIR="$RFT_RUNTIME_DIR/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$XDG_CONFIG_HOME/wandb"
export WANDB_DATA_DIR="$RFT_RUNTIME_DIR/wandb-data"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# rLLM/Ray advertise the node's LAN IP for both the gateway and vLLM workers.
# Those local requests must bypass download proxies, including in child processes.
IFS=',' read -r -a _rft_proxy_exclusions <<< "${NO_PROXY:-},${no_proxy:-}"
read -r -a _rft_local_ips <<< "$(hostname -I 2>/dev/null || true)"
_rft_no_proxy=""
for _rft_host in localhost 127.0.0.1 ::1 "$(hostname)" "${_rft_local_ips[@]}" "${_rft_proxy_exclusions[@]}"; do
  [[ -n "$_rft_host" ]] || continue
  if [[ ",$_rft_no_proxy," != *",$_rft_host,"* ]]; then
    _rft_no_proxy="${_rft_no_proxy:+$_rft_no_proxy,}$_rft_host"
  fi
done
export NO_PROXY="$_rft_no_proxy"
export no_proxy="$NO_PROXY"
unset _rft_proxy_exclusions _rft_local_ips _rft_no_proxy _rft_host

if [[ ":$PATH:" != *":$RFT_RUNTIME_DIR/bin:"* ]]; then
  export PATH="$RFT_RUNTIME_DIR/bin:$PATH"
fi
mkdir -p "$RFT_RUNTIME_DIR/bin" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$HF_HOME" "$WANDB_DIR" || return 1
