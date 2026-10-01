#!/usr/bin/env bash
# CTRL inference on (a subset of) Waymo validation with the authors' FSD detections and checkpoints.
#   CLS=vehicle|pedestrian|cyclist  bash run_val_pipeline.sh
# Steps: finalize tables -> FSD baseline eval -> ImmortalTracker (keep10) -> extend_tracks (backward)
#        -> generate_track_input -> CTRL test (official ckpt) -> remove_empty -> eval each stage.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../scripts/env.sh"
CLS="${CLS:-vehicle}"
SST="$CTRL_ROOT/SST"
TRK="$CTRL_ROOT/ImmortalTracker-for-CTRL"
RES="$CTRL_ROOT/resources"
export TF_CPP_MIN_LOG_LEVEL=3
NPROC="${NPROC:-4}"

case "$CLS" in
  vehicle)    SHORT=vehicle; DET=fsd_base_vehicle_val; CKPT=ctrl_veh_epoch_24; CFG=ctrl_veh_24e; YAML_SRC=fsd_base_vehicle;    YAML=fsd_base_vehicle ;;
  pedestrian) SHORT=ped;     DET=fsd_base_ped_val;     CKPT=ped_epoch_24;      CFG=ctrl_ped_24e; YAML_SRC=fsd_base_pedestrian; YAML=fsd_base_ped ;;
  cyclist)    SHORT=cyc;     DET=fsd_base_cyc_val;     CKPT=cyc_epoch_12;      CFG=ctrl_cyc_12e; YAML_SRC=fsd_base_cyc;        YAML=fsd_base_cyc ;;
  *) echo "unknown CLS=$CLS"; exit 1 ;;
esac
OUT="$SST/data/ctrl_runs/$CLS"
mkdir -p "$OUT"
EVAL="$CTRL_ROOT/bin/compute_detection_metrics_main"
GT="$SST/data/waymo/waymo_format/gt.bin"
summary() {  # name bin -> one line with L1/L2 mAP/mAPH for the class
  "$EVAL" "$2" "$GT" > "$OUT/$1.txt" 2>&1 || true
  key=$(echo "$CLS" | tr a-z A-Z)
  echo "$1: $(grep -a "OBJECT_TYPE_TYPE_${key}_LEVEL_[12]" "$OUT/$1.txt" | sed 's/OBJECT_TYPE_TYPE_//' | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
}

echo "== [1] finalize lookup tables / gt.bin / subset detections"
python "$HERE/finalize_val.py" --sst "$SST" --sst-data "$SST/data" --poses "$RES/poses.pkl" --det "$RES/$DET.bin"
: > "$OUT/summary.txt"
summary fsd_baseline "$SST/data/ctrl_bins/validation/$DET.bin"

echo "== [2] ImmortalTracker"
cd "$TRK"
python preparedata/waymo/detection.py --file_name "$SST/data/ctrl_bins/validation/$DET.bin" --name "$DET" \
  --det_folder ./data/waymo --data_folder ./data/waymo --split validation
python main_waymo.py --name immortal --det_name "$DET" --config_path configs/waymo_configs/immortal_for_ctrl_keep10.yaml \
  --process "$NPROC" --det_data_folder ./data/waymo --obj_type "$CLS" --split validation
python evaluation/waymo/pred_bin.py --name immortal --det_name "$DET" --obj_types "$CLS" \
  --config_path configs/waymo_configs/immortal_for_ctrl_keep10.yaml --split validation --no-eval
cp "mot_results/waymo/validation/immortal_$DET/bin/$CLS/pred.bin" "$OUT/track.bin"
summary track_keep10 "$OUT/track.bin"

echo "== [3] backward extension"
cd "$SST"
sed "s#^bin_path:.*#bin_path: $OUT/track.bin#" tools/ctrl/data_configs/extend.yaml > "$OUT/extend.yaml"
python tools/ctrl/extend_tracks.py "$OUT/extend.yaml" > "$OUT/extend.log" 2>&1 || true
summary track_bi_extend "$OUT/track_extend.bin"

echo "== [4] track input"
sed -e "s#^val_bin_path:.*#val_bin_path: $OUT/track_extend.bin#" -e "s#^exist_ok:.*#exist_ok: True#" -e "s#^split:.*#split: val#" \
  "tools/ctrl/data_configs/$YAML_SRC.yaml" > "$OUT/$YAML.yaml"
python tools/ctrl/generate_track_input.py "$OUT/$YAML.yaml" --process "$NPROC"

echo "== [5] CTRL inference ($CFG, official checkpoint)"
python tools/test.py "configs/ctrl/$CFG.py" "$RES/$CKPT.pth" --eval waymo \
  --cfg-options data.test.samples_per_gpu="${BATCH:-4}" data.workers_per_gpu=4 \
  --eval-options "pklfile_prefix=$OUT/ctrl" > "$OUT/test.log" 2>&1 || true
summary ctrl "$OUT/ctrl.bin"

echo "== [6] remove empty boxes"
python tools/ctrl/remove_empty.py --bin-path "$OUT/ctrl.bin" --split training --type "$CLS" --process "$NPROC" > "$OUT/remove_empty.log" 2>&1 || true
summary ctrl_no_empty "$OUT/ctrl_wo_empty_right.bin"

echo "== summary ($CLS) -> $OUT/summary.txt"
cat "$OUT/summary.txt"
