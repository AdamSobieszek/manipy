#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-fob}"
FOB_DIR="${2:-$PWD/FOB}"
OUT_DIR="${3:-$PWD/outputs-mini}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

EXP="${FOB_DIR}/experiment_traversal_mini.yaml"
cat > "${EXP}" <<'YAML'
task:
  - name: mnist
    max_epochs: 3
  - name: classification_small
    max_epochs: 3
  - name: graph_tiny
    max_epochs: 3
optimizer:
  name: traversal_adamw
  learning_rate: [1.0e-3, 3.0e-4]
  weight_decay: [0.0, 1.0e-2, 1.0e-3]
  traversal_mode: [none, batch, full]
  moment1_source: [trav]
  moment2_source: [trav, grad]
engine:
  seed: [1, 2]
  output_dir: "<<OUT_DIR>>"
  devices: 1
  train: true
  test: true
  plot: true
YAML
sed -i.bak "s#<<OUT_DIR>>#${OUT_DIR}#g" "${EXP}"

echo "[TraversalAdamW] Preparing datasets ..."
python -m pytorch_fob.dataset_setup "${EXP}"

echo "[TraversalAdamW] Running experiment ..."
python -m pytorch_fob.run_experiment "${EXP}"
