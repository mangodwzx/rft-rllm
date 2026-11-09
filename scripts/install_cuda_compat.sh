#!/usr/bin/env bash
# Extract NVIDIA userspace libraries in this repo; never install a system driver.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
DEST="$ROOT/vendor/cuda-compat-13-0"
PACKAGE="cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb"
SHA256="b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035"
URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$PACKAGE"
[[ "$(uname -m)" == x86_64 ]] || { echo 'This compatibility package requires x86_64.' >&2; exit 1; }
command -v dpkg-deb >/dev/null
mkdir -p "$DEST"
if [[ ! -f "$DEST/$PACKAGE" ]]; then
  curl --fail --location --connect-timeout 30 --max-time 900 \
    --speed-limit 1024 --speed-time 60 --retry 5 --retry-delay 3 \
    "$URL" --output "$DEST/$PACKAGE.partial"
  printf '%s  %s\n' "$SHA256" "$DEST/$PACKAGE.partial" | sha256sum --check -
  mv "$DEST/$PACKAGE.partial" "$DEST/$PACKAGE"
fi
printf '%s  %s\n' "$SHA256" "$DEST/$PACKAGE" | sha256sum --check -
dpkg-deb -x "$DEST/$PACKAGE" "$DEST"
test -f "$DEST/usr/local/cuda-13.0/compat/libcuda.so.1"
printf 'Local CUDA compatibility libraries: %s\n' "$DEST"
