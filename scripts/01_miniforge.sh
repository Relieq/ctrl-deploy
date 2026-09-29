#!/usr/bin/env bash
# Install Miniforge (conda + mamba) into $HOME/miniforge3
set -eo pipefail
PREFIX="${CONDA_PREFIX_DIR:-$HOME/miniforge3}"
if [ ! -x "$PREFIX/bin/conda" ]; then
  apt-get update -y && apt-get install -y wget git build-essential ninja-build unzip
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$PREFIX"
  "$PREFIX/bin/conda" init bash
fi
"$PREFIX/bin/conda" --version
