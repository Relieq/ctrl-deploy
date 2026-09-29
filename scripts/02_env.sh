#!/usr/bin/env bash
# Create conda env "ctrl": python 3.10, CUDA 12.8 toolkit (nvcc), gcc/g++ 12, torch 2.7.1+cu128
set -eo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
ENV=ctrl
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n $ENV -c conda-forge -c nvidia/label/cuda-12.8.1 \
    python=3.10 cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-libraries-dev=12.8 cuda-cccl=12.8 \
    gcc_linux-64=12 gxx_linux-64=12 ninja cmake
fi
conda activate $ENV
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device='cuda'); print('matmul ok', (x @ x).sum().item() != 0)
PY
nvcc --version | tail -2
