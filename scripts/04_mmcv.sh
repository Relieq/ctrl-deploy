#!/usr/bin/env bash
# Build mmcv-full 1.7.2 from source for sm_120 (torch 2.x needs C++17)
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$CTRL_ROOT/mmcv"
sed -i 's/-std=c++14/-std=c++17/g' setup.py
pip install "setuptools==69.5.1" wheel
pip install -r requirements/runtime.txt
# torch>=2: torch.nn.parallel._get_stream expects a torch.device, not an int
sed -i 's/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]/' mmcv/parallel/_functions.py
MMCV_WITH_OPS=1 pip install -v --no-build-isolation -e . > "$CTRL_ROOT/build_mmcv.log" 2>&1 || { tail -40 "$CTRL_ROOT/build_mmcv.log"; exit 1; }
python -c "import mmcv, torch; from mmcv.ops import nms, RoIAlign; import mmcv.ops as o; b=torch.rand(10,4,device='cuda'); b[:,2:]+=b[:,:2]; print('mmcv', mmcv.__version__, nms(b, torch.rand(10,device='cuda'), 0.5)[1].shape)"
