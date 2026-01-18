#!/bin/bash
set -e

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

echo "Building with VERSION=${VERSION}"
docker build --build-arg VERSION="${VERSION}" -t wyoming-xtts:local .