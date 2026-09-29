#!/usr/bin/env bash
# Stage 3: 1-epoch CTRL training + inference on the synthetic tracklets (single GPU, no launcher).
set -eo pipefail
source "$(dirname "$0")/../scripts/env.sh"
SMOKE="${SMOKE:-$CTRL_ROOT/smoke}"
cd "$SMOKE/sst"
cp "$(dirname "$0")/ctrl_veh_smoke.py" "$CTRL_ROOT/SST/configs/ctrl/ctrl_veh_smoke.py"
python tools/train.py configs/ctrl/ctrl_veh_smoke.py --work-dir work_dirs/ctrl_veh_smoke --no-validate
mkdir -p results
python tools/test.py configs/ctrl/ctrl_veh_smoke.py work_dirs/ctrl_veh_smoke/latest.pth \
  --eval waymo --options "pklfile_prefix=./results/ctrl_smoke" || echo "(metric binary step failed)"
ls -la results
