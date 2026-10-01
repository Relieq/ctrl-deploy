#!/usr/bin/env bash
# Download the authors' public CTRL resources needed for Waymo *validation* inference (Google Drive).
# Training files (train_gt.bin, fsd*_train*.bin) are skipped; add them to FILES if you want to train.
set -eo pipefail
source "$(dirname "$0")/env.sh"
pip install -q gdown
OUT="$CTRL_ROOT/resources"
mkdir -p "$OUT" && cd "$OUT"

FILES=(
  "1Q5Gu72KzZz23M6X_iGzvR6HfPI4h4F2s ctrl_veh_epoch_24.pth"
  "1hinxg_I8_Irq39u93cRUdGHfs7AgYhYZ fsd_base_vehicle_val.bin"
  "1Y2XM2vadTnksEXIA3RNLUqXX0Ar2BGfW poses.pkl"
  "1ZcWktcWTrWWEwxakqetJCCfWXBjeIErD ped_epoch_24.pth"
  "1f7venIvJqD0-cGaNvAJfaOXv5RmjqhnJ fsd_base_ped_val.bin"
  "13UlieBi21Hx3vdBR1Dd2Oj_pD_YTIM9j cyc_epoch_12.pth"
  "1qp6pTfGGwX2iCafYjehwbic-7eHhZz9b fsd_base_cyc_val.bin"
  "1Z518w8RhN4tSJhGuMg1-CPQy5zLETs5v reference/best_val_in_paper.bin"
  "1gJpDZ43SGYs9HS5hgHvKpSAlUl19Spqq reference/cyc_result_val_bi_ext_no_empty.bin"
  "1aiJINgo34rHb4PUcQhn1Y-TH3HYMPqnC reference/cyc_result_val_bi_ext_no_empty.txt"
)
for entry in "${FILES[@]}"; do
  id="${entry%% *}"; name="${entry#* }"
  mkdir -p "$(dirname "$name")"
  [ -s "$name" ] || gdown -q "$id" -O "$name"
done
ls -la "$OUT" "$OUT/reference"
