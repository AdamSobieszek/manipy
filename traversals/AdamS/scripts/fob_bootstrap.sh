#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
ENV_NAME="${2:-fob}"

echo "[FOB] Cloning into ${ROOT}/FOB ..."
mkdir -p "${ROOT}"
cd "${ROOT}"
if [ ! -d FOB ]; then
  git clone https://github.com/automl/FOB.git
fi
cd FOB

echo "[FOB] Creating conda env ${ENV_NAME} (Python 3.10) ..."
conda create -y -n "${ENV_NAME}" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[FOB] Installing requirements and package ..."
pip install -r requirements.txt
pip install -e .

echo "[FOB] Note: if mmcv fails to install correctly, follow the README hint to install the exact version."
echo "[FOB] Sanity check: printing help"
python -c "import pytorch_fob, sys; print('pytorch_fob OK')"

echo "[FOB] Done."
