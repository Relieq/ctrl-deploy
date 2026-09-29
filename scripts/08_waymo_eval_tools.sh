#!/usr/bin/env bash
# Build the official Waymo C++ metric tools (compute_detection_metrics_main, compute_tracking_metrics_main)
# with Bazel and put them where SST / ImmortalTracker expect them.
set -eo pipefail
source "$(dirname "$0")/env.sh"
WOD_TAG="${WOD_TAG:-v1.6.1}"
cd "$CTRL_ROOT"
[ -d waymo-open-dataset ] || git clone --depth 1 -b "$WOD_TAG" https://github.com/waymo-research/waymo-open-dataset.git
if ! command -v bazelisk >/dev/null; then
  wget -q https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O /usr/local/bin/bazelisk
  chmod +x /usr/local/bin/bazelisk
fi
cd waymo-open-dataset/src
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
bazelisk build -c opt \
  //waymo_open_dataset/metrics/tools:compute_detection_metrics_main \
  //waymo_open_dataset/metrics/tools:compute_tracking_metrics_main
BIN=bazel-bin/waymo_open_dataset/metrics/tools
mkdir -p "$CTRL_ROOT/bin"
cp -f $BIN/compute_detection_metrics_main $BIN/compute_tracking_metrics_main "$CTRL_ROOT/bin/"
# SST evaluate() calls ./mmdet3d/core/evaluation/waymo_utils/compute_detection_metrics_main (relative to cwd = SST root)
ln -sfn "$CTRL_ROOT/bin/compute_detection_metrics_main" "$CTRL_ROOT/SST/mmdet3d/core/evaluation/waymo_utils/compute_detection_metrics_main"
# ImmortalTracker pred_bin.py calls ./compute_tracking_metrics_main_detail (relative to tracker root)
ln -sfn "$CTRL_ROOT/bin/compute_tracking_metrics_main" "$CTRL_ROOT/ImmortalTracker-for-CTRL/compute_tracking_metrics_main_detail"
ls -la "$CTRL_ROOT/bin"
