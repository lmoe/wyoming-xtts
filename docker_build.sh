#!/bin/bash
set -e

# CUDA variant: cu126 (Pascal-Ada, default) or cu128 (Volta-Blackwell)
CUDA_VARIANT="${1:-cu126}"

case "$CUDA_VARIANT" in
    cu126)
        CUDA_IMAGE="nvidia/cuda:12.6.3-devel-ubuntu22.04"
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        TORCH_ARCH="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
        ;;
    cu128)
        CUDA_IMAGE="nvidia/cuda:12.8.1-devel-ubuntu22.04"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_ARCH="7.0;7.5;8.0;8.6;8.9;9.0;12.0"
        ;;
    *)
        echo "Unknown CUDA variant: $CUDA_VARIANT (use cu126 or cu128)"
        exit 1
        ;;
esac

if ! command -v ruff &>/dev/null || ! command -v mypy &>/dev/null; then
    echo "Installing dev tools..."
    pip install -q ruff mypy types-requests
fi

echo "Running lint checks..."
ruff check --fix .
ruff format .

echo "Running type checks..."
mypy wyoming_xtts/

if VERSION=$(git describe --tags --exact-match 2>/dev/null); then
    VERSION="${VERSION#v}"  # Strip 'v' prefix
else
    SHA=$(git rev-parse --short HEAD)
    VERSION="0.0.0.dev0+${SHA}"
fi

echo "Building ${CUDA_VARIANT} with VERSION=${VERSION}"
docker build \
    --build-arg VERSION="${VERSION}" \
    --build-arg CUDA_IMAGE="${CUDA_IMAGE}" \
    --build-arg TORCH_INDEX_URL="${TORCH_INDEX}" \
    --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}" \
    -t "lmo3/wyoming-xtts:${CUDA_VARIANT}" .
