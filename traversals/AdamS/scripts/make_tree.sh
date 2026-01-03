#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
mkdir -p "${ROOT}/our_opt" "${ROOT}/scripts" "${ROOT}/outputs-mini" "${ROOT}/outputs-full"
echo "[Tree] Created: our_opt/, scripts/, outputs-mini/, outputs-full/"
