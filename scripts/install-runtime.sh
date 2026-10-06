#!/usr/bin/env bash
# Install one of the locked CPython 3.12 runtime profiles.
# Supports Windows (Git Bash) and Linux x86_64 profile locks.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/install-runtime.sh [--torch-backend cpu|cu126] [--python PATH]

Examples:
  bash scripts/install-runtime.sh --torch-backend cpu --python python3
  bash scripts/install-runtime.sh --torch-backend cu126 --python .venv/Scripts/python.exe

The committed locks target Windows/Linux x86_64 + CPython 3.12. Re-resolve
the requirements for other operating systems, Python versions, or CUDA backends.
EOF
}

torch_backend="cpu"
python_bin="python3"

while (($# > 0)); do
    case "$1" in
        --torch-backend|-b)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            torch_backend="$2"
            shift 2
            ;;
        --python|-p)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            python_bin="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$torch_backend" in
    cpu)
        torch_index="https://download.pytorch.org/whl/cpu"
        ;;
    cu126)
        torch_index="https://download.pytorch.org/whl/cu126"
        ;;
    *)
        echo "Unsupported --torch-backend: $torch_backend (expected cpu or cu126)" >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
requirements_root="$project_root/requirements"
python_platform="$("$python_bin" -c 'import sys; print(sys.platform)')"
python_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_arch="$("$python_bin" -c 'import platform; print(platform.machine().lower())')"

case "$python_platform" in
    win32)
        profile_suffix=""
        expected_arch="amd64"
        ;;
    linux)
        profile_suffix="-linux"
        expected_arch="x86_64"
        ;;
    *)
        echo "No committed runtime profile exists for platform: $python_platform" >&2
        echo "Generate a platform-specific base and Torch lock before installing there." >&2
        exit 1
        ;;
esac

if [[ "$python_version" != "3.12" || "$python_arch" != "$expected_arch" ]]; then
    echo "Selected interpreter is $python_platform/$python_arch Python $python_version." >&2
    echo "Committed locks require $python_platform/$expected_arch Python 3.12." >&2
    exit 1
fi

base_lock="$requirements_root/base${profile_suffix}.lock"
torch_lock="$requirements_root/torch-${torch_backend}${profile_suffix}.lock"

[[ -f "$base_lock" ]] || { echo "Missing dependency lock: $base_lock" >&2; exit 1; }
[[ -f "$torch_lock" ]] || { echo "Missing PyTorch profile lock: $torch_lock" >&2; exit 1; }

echo "Installing locked base dependencies..."
"$python_bin" -m pip install --require-hashes -r "$base_lock"

echo "Installing locked PyTorch profile: $torch_backend..."
# PyTorch wheels use the official PyTorch index. Torch's ordinary Python
# dependencies remain resolvable from PyPI; the `+cu126` requirement in the
# lock keeps pip from replacing the requested CUDA build with a CPU wheel.
"$python_bin" -m pip install --require-hashes --extra-index-url "$torch_index" -r "$torch_lock"

"$python_bin" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

if [[ "$torch_backend" == "cu126" ]]; then
    echo "CUDA wheel installed. cuda_available must be True before selecting CUDA in the application."
fi
