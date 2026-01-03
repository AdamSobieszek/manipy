#!/usr/bin/env bash
set -euo pipefail

FOB_DIR="${1:-$PWD/FOB}"
SRC_DIR="${2:-$PWD/our_opt}"

if [ ! -d "${FOB_DIR}" ]; then
  echo "FOB not found at ${FOB_DIR}"
  exit 1
fi
if [ ! -d "${SRC_DIR}" ]; then
  echo "our_opt not found at ${SRC_DIR}"
  exit 1
fi

TARGET="${FOB_DIR}/pytorch_fob/optimizers/traversal_adamw"
mkdir -p "${TARGET}"
cp -f "${SRC_DIR}/traversal_adamw_impl.py" "${TARGET}/traversal_adamw_impl.py"
cp -f "${SRC_DIR}/traversal_adamw_configure.py" "${TARGET}/configure.py"
cp -f "${SRC_DIR}/default.yaml" "${TARGET}/default.yaml"

if [ ! -f "${TARGET}/__init__.py" ]; then
  echo "from .configure import configure_optimizers  # re-export" > "${TARGET}/__init__.py"
fi

echo "[TraversalAdamW] Installed at ${TARGET}"
