#!/usr/bin/env bash
# Build the pinned, unmodified NVIDIA SM-copy bandwidth reference.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: build_nvbandwidth.sh NEW_SOURCE_DIR [CUDA_ARCH=80]' >&2
    exit 2
fi
nvbw_source=$1
nvbw_arch=${2:-80}
nvbw_commit=82fc4e8c6afa0babb8687793678f615b3b8d793e
if [[ -e "$nvbw_source" ]]; then
    echo "Use a new directory: $nvbw_source" >&2
    exit 2
fi
git clone --no-checkout https://github.com/NVIDIA/nvbandwidth.git "$nvbw_source"
git -C "$nvbw_source" checkout --detach "$nvbw_commit"
cmake -S "$nvbw_source" -B "$nvbw_source/build" \
    -DCMAKE_CUDA_ARCHITECTURES="$nvbw_arch" -DCMAKE_BUILD_TYPE=Release
cmake --build "$nvbw_source/build" -j "${BUILD_JOBS:-8}"
