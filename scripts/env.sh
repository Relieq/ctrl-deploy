# Source this file: activates env "ctrl" and sets build variables for sm_120
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ctrl
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export CTRL_ROOT="${CTRL_ROOT:-$HOME/CTRL}"
export PIP_ROOT_USER_ACTION=ignore
