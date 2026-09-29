#!/usr/bin/env bash
# Port + build SST (mmdet3d 0.15 fork) CUDA ops for sm_120, install runtime deps.
set -eo pipefail
source "$(dirname "$0")/env.sh"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
cd "$CTRL_ROOT/SST"
python "$SCRIPTS/sst_port_torch2.py"
python "$SCRIPTS/tools_patch.py" "$CTRL_ROOT"
pip install -v --no-build-isolation --no-deps -e . > "$CTRL_ROOT/build_sst.log" 2>&1 || { grep -n "error" "$CTRL_ROOT/build_sst.log" | head -40; exit 1; }
# runtime deps (upstream pins numba 0.48 / numpy<1.20, which cannot coexist with torch 2.x)
pip install "numba==0.58.1" "numpy==1.23.5" nuscenes-devkit lyft_dataset_sdk "networkx<3" plyfile \
  "scikit-image<0.22" tensorboard "trimesh>=2.35.39,<2.35.40" "shapely<2" "pyyaml" filterpy
pip install "numpy==1.23.5" "setuptools==69.5.1"
