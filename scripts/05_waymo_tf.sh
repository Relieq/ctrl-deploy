#!/usr/bin/env bash
# TensorFlow (CPU) + Waymo Open Dataset devkit in the same env as torch.
# WOD 1.6.4 hard-pins ~20 heavy packages; we only need its protos/utils, so install with --no-deps.
set -eo pipefail
source "$(dirname "$0")/env.sh"
pip install "numpy==1.23.5" "yapf==0.40.1"
pip install tensorflow-cpu==2.12.0
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
pip install "absl-py==1.4.0" "immutabledict==2.2.0" "dacite==1.8.1" "einsum==0.3.0"
# TF 2.12 pins typing_extensions<4.6 but torch 2.7 needs >=4.10; runtime works with the newer one.
pip install "typing_extensions>=4.10" "setuptools==69.5.1"
# WOD 1.6.4 utils call ParseFromString(bytearray(...)), which protobuf>=4 rejects
pip install "protobuf==3.20.3"
python - <<'PY'
import tensorflow as tf, torch
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils, box_utils
print('tf', tf.__version__, 'torch', torch.__version__, torch.cuda.is_available(), 'wod ok')
PY
