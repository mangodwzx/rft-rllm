#!/usr/bin/env bash
# Prepare and check the pip-installed CUDA runtime before expensive builds.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Project environment is missing.' >&2; exit 1; }
command -v g++ >/dev/null

CUDA_LIB_DIR="$("$PYTHON" - "$ROOT" <<'PY'
import filecmp
from pathlib import Path
import sys
import sysconfig

root = Path(sys.argv[1]).resolve()
library_dir = (Path(sysconfig.get_path("purelib")) / "nvidia/cu13/lib").resolve()
if not library_dir.is_relative_to(root):
    raise SystemExit(f"Refusing to modify CUDA libraries outside this repository: {library_dir}")
runtime = library_dir / "libcudart.so.13"
link = library_dir / "libcudart.so"
if not runtime.is_file() or not runtime.resolve().is_relative_to(root):
    raise SystemExit(f"Project CUDA 13 runtime is missing or external: {runtime}")
if link.exists() or link.is_symlink():
    if not link.is_file() or not link.resolve().is_relative_to(root):
        raise SystemExit(f"Conflicting libcudart linker file; preserve and inspect it: {link}")
    if not link.samefile(runtime) and not filecmp.cmp(link, runtime, shallow=False):
        raise SystemExit(f"Conflicting libcudart linker file; preserve and inspect it: {link}")
else:
    # NVIDIA's runtime wheel contains the SONAME file but no unversioned name
    # for the linker's -lcudart lookup. Keep the alias inside this venv.
    link.symlink_to(runtime.name)
print(library_dir)
PY
)"

# Link a tiny shared library without importing CUDA or touching a GPU. Fail
# here instead of discovering an unusable -lcudart after a long kernel build.
CHECK_DIR="$(mktemp -d "$TMPDIR/rft-cudart-link-XXXXXX")"
trap 'rm -rf -- "$CHECK_DIR"' EXIT
g++ -x c++ - -shared -fPIC -Wl,--no-undefined \
  -L"$CUDA_LIB_DIR" -lcudart -o "$CHECK_DIR/probe.so" <<'CPP'
extern "C" int cudaRuntimeGetVersion(int*);
extern "C" int rft_cuda_link_probe(int* version) {
    return cudaRuntimeGetVersion(version);
}
CPP
printf 'CUDA runtime linker check passed: %s/libcudart.so\n' "$CUDA_LIB_DIR"
