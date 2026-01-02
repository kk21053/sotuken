#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD_PATH="${1:-"$ROOT_DIR/worlds/sotuken_world.wbt"}"

exec webots "$WORLD_PATH"
