#!/usr/bin/env bash
# Write the current upstream modifications as reference diffs into ../patches/
source "$(dirname "$0")/env.sh"
OUT="$(cd "$(dirname "$0")/.." && pwd)/patches"
mkdir -p "$OUT"
for r in SST ImmortalTracker-for-CTRL mmcv; do
  git -C "$CTRL_ROOT/$r" diff > "$OUT/$r.patch"
  echo "$r: $(git -C "$CTRL_ROOT/$r" diff --stat | tail -1)"
done
