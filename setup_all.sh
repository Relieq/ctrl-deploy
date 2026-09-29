#!/usr/bin/env bash
# Full environment setup inside WSL2 Ubuntu. Run as: bash setup_all.sh
set -eo pipefail
S="$(cd "$(dirname "$0")/scripts" && pwd)"
bash "$S/01_miniforge.sh"
bash "$S/02_env.sh"
bash "$S/03_clone.sh"
bash "$S/04_mmcv.sh"
bash "$S/05_waymo_tf.sh"
bash "$S/06_openmmlab_sparse.sh"
bash "$S/07_sst.sh"
bash "$S/08_waymo_eval_tools.sh"

source "$S/env.sh"
cd "$CTRL_ROOT/SST" && python "$S/check_sst.py"
echo "Setup finished. Run smoke tests with: bash $(dirname "$S")/smoke_test/run_all.sh"
