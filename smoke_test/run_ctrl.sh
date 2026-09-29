#!/usr/bin/env bash
# Stage 2: SST/CTRL tools on the synthetic tracker output. Workdir: $SMOKE/sst (holds ./data).
# The synthetic validation set doubles as "training" data so the train path runs too.
set -eo pipefail
source "$(dirname "$0")/../scripts/env.sh"
SMOKE="${SMOKE:-$CTRL_ROOT/smoke}"
SST="$CTRL_ROOT/SST"
export TF_CPP_MIN_LOG_LEVEL=2
cd "$SMOKE/sst"
ln -sfn "$SST/configs" configs
ln -sfn "$SST/tools" tools
# eval binaries are invoked by relative path mmdet3d/core/evaluation/waymo_utils/... from the workdir
mkdir -p mmdet3d/core/evaluation/waymo_utils
ln -sfn "$CTRL_ROOT/bin/compute_detection_metrics_main" mmdet3d/core/evaluation/waymo_utils/compute_detection_metrics_main
TRK_BIN="$SMOKE/trk/mot_results/waymo/validation/immortal_fsd_synth/bin/pred_merged.bin"
mkdir -p data/ctrl_bins cfg
cp "$TRK_BIN" data/ctrl_bins/trk_val.bin

echo "== generate_train_gt_bin / extract_poses"
python tools/ctrl/generate_train_gt_bin.py --data-folder data/waymo/waymo_format/validation --output data/waymo/waymo_format/train_gt.bin
python tools/ctrl/extract_poses.py

echo "== extend_tracks (backward)"
sed "s#^bin_path:.*#bin_path: ./data/ctrl_bins/trk_val.bin#" tools/ctrl/data_configs/extend.yaml > cfg/extend.yaml
python tools/ctrl/extend_tracks.py cfg/extend.yaml || echo "(extend_tracks eval step failed - expected without compute_detection_metrics_main)"
ls data/ctrl_bins

echo "== generate_track_input (training + val)"
for split in training val; do
  sed -e "s#^bin_path:.*#bin_path: ./data/ctrl_bins/trk_val_extend.bin#" \
      -e "s#^val_bin_path:.*#val_bin_path: ./data/ctrl_bins/trk_val_extend.bin#" \
      -e "s#^exist_ok:.*#exist_ok: True#" -e "s#^split:.*#split: $split#" \
      tools/ctrl/data_configs/fsd_base_vehicle.yaml > cfg/fsd_base_vehicle.yaml
  python tools/ctrl/generate_track_input.py cfg/fsd_base_vehicle.yaml --process 2
done
echo "== generate_candidates (training)"
sed -i "s#^split:.*#split: training#" cfg/fsd_base_vehicle.yaml
python tools/ctrl/generate_candidates.py cfg/fsd_base_vehicle.yaml --process 2
ls -la data/waymo/tracklet_data
