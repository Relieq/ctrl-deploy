#!/usr/bin/env bash
# End-to-end smoke test on synthetic data: tracker -> CTRL data prep -> 1-epoch train -> inference -> eval.
set -eo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
source "$T/../scripts/env.sh"
export SMOKE="${SMOKE:-$CTRL_ROOT/smoke}"
rm -rf "$SMOKE"
python "$T/make_synthetic.py" --sst "$CTRL_ROOT/SST" --out "$SMOKE"
bash "$T/run_tracker.sh"
bash "$T/run_ctrl.sh"
bash "$T/run_train_test.sh"
cd "$SMOKE"
python "$T/bin_stats.py" sst/data/waymo/waymo_format/gt.bin det/fsd_synth_val.bin \
  trk/mot_results/waymo/validation/immortal_fsd_synth/bin/pred_merged.bin \
  sst/data/ctrl_bins/trk_val_extend.bin sst/results/ctrl_smoke.bin
