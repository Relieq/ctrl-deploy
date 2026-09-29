#!/usr/bin/env bash
# Clone source repos into $CTRL_ROOT (ext4 inside WSL) at the commits this setup was validated with.
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$CTRL_ROOT" && cd "$CTRL_ROOT"

clone_at() {  # url dir commit
  [ -d "$2" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
  echo "$2 $(git -C "$2" rev-parse --short HEAD)"
}
clone_at https://github.com/tusen-ai/SST.git SST 7c95376
clone_at https://github.com/Abyssaledge/ImmortalTracker-for-CTRL.git ImmortalTracker-for-CTRL 03f25eb
clone_at https://github.com/Abyssaledge/TorchEx.git TorchEx f13da8a
[ -d mmcv ] || git clone -b v1.7.2 --depth 1 https://github.com/open-mmlab/mmcv.git
