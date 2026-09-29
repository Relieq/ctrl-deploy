#!/usr/bin/env bash
# Stage 1: ImmortalTracker on the synthetic set. Workdir: $SMOKE/trk (keeps repo clean).
set -eo pipefail
source "$(dirname "$0")/../scripts/env.sh"
SMOKE="${SMOKE:-$CTRL_ROOT/smoke}"
TRK="$CTRL_ROOT/ImmortalTracker-for-CTRL"
export PYTHONPATH="$TRK:$PYTHONPATH" TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$SMOKE/trk" && cd "$SMOKE/trk"
ln -sfn "$TRK/configs" configs
# pred_bin.py evaluates with ./compute_tracking_metrics_main_detail <pred> ./gt.bin
ln -sfn "$CTRL_ROOT/bin/compute_tracking_metrics_main" compute_tracking_metrics_main_detail
ln -sfn "$SMOKE/sst/data/waymo/waymo_format/gt.bin" gt.bin
WFMT="$SMOKE/sst/data/waymo/waymo_format"
python "$TRK/preparedata/waymo/time_stamp.py" --data_folder "$WFMT"
python "$TRK/preparedata/waymo/ego_info.py" --data_folder "$WFMT" --process 2
python "$TRK/preparedata/waymo/detection.py" --file_name "$SMOKE/det/fsd_synth_val.bin" --name fsd_synth \
  --det_folder ./detection_data/processed --split validation
python "$TRK/main_waymo.py" --name immortal --det_name fsd_synth --config_path configs/waymo_configs/immortal_for_ctrl_keep10.yaml \
  --process 2 --det_data_folder ./detection_data/processed --obj_type vehicle --split validation
python "$TRK/evaluation/waymo/pred_bin.py" --name immortal --det_name fsd_synth \
  --config_path configs/waymo_configs/immortal_for_ctrl_keep10.yaml --split validation
grep -m4 "MOTA\|OVERALL\|TYPE_VEHICLE_LEVEL_2" -r mot_results/waymo/validation/immortal_fsd_synth/bin/ || true
