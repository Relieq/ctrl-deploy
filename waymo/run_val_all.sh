#!/usr/bin/env bash
# Run run_val_pipeline.sh for the given classes (default: all three), one log per class in $CTRL_ROOT.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CTRL_ROOT:-$HOME/CTRL}"
for c in "${@:-vehicle pedestrian cyclist}"; do
  for cls in $c; do
    CLS=$cls bash "$HERE/run_val_pipeline.sh" > "$ROOT/pipeline_$cls.log" 2>&1
    echo PIPELINE_DONE >> "$ROOT/pipeline_$cls.log"
  done
done
