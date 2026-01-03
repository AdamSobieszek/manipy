#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-fob}"
EXP_YAML="${2:-$PWD/experiment_traversal_mini.yaml}"
FOB_DIR="${3:-$PWD/FOB}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${FOB_DIR}"
python -m pytorch_fob.dataset_setup "${EXP_YAML}"
