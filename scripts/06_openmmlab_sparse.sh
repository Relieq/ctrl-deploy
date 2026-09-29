#!/usr/bin/env bash
# mmdet 2.x / mmseg 0.x (last releases compatible with mmcv 1.x), torch_scatter, spconv2, TorchEx
set -eo pipefail
source "$(dirname "$0")/env.sh"
pip install "mmdet==2.28.2" "mmsegmentation==0.30.0" "numpy==1.23.5"
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install spconv-cu126 ipdb
cd "$CTRL_ROOT/TorchEx"
pip install -v --no-build-isolation -e . > "$CTRL_ROOT/build_torchex.log" 2>&1 || { grep -n "error" "$CTRL_ROOT/build_torchex.log" | head -30; exit 1; }
python "$(dirname "$0")/check_sparse.py"
