#!/usr/bin/env bash
# Optional: longer training on the synthetic tracklets to confirm the model actually learns (mAP should rise).
set -eo pipefail
source "$(dirname "$0")/../scripts/env.sh"
SMOKE="${SMOKE:-$CTRL_ROOT/smoke}"
EPOCHS="${EPOCHS:-40}"
cd "$SMOKE/sst"
python tools/train.py configs/ctrl/ctrl_veh_smoke.py --work-dir work_dirs/ctrl_veh_learn --no-validate \
  --cfg-options runner.max_epochs=$EPOCHS log_config.interval=23 checkpoint_config.interval=$EPOCHS 2>&1 \
  | grep -a "Epoch \[\(1\|$EPOCHS\)\]\[20" || true
python tools/test.py configs/ctrl/ctrl_veh_smoke.py work_dirs/ctrl_veh_learn/latest.pth \
  --eval waymo --options "pklfile_prefix=./results/ctrl_learn" 2>&1 | grep -a "Vehicle/L1 mAP'" | head -1 \
  | python -c "import sys,ast; d=ast.literal_eval(sys.stdin.read().strip()); print({k: round(v, 4) for k, v in d.items() if k.startswith('Vehicle')})"
