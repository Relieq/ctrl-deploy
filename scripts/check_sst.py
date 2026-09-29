"""Smoke test: SST/mmdet3d CUDA ops on the CTRL path + build the CTRL TrackletDetector."""
import sys
import torch
import mmcv, mmdet, mmseg, mmdet3d
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmseg', mmseg.__version__, 'mmdet3d', mmdet3d.__version__)
import mmdet3d.ops  # imports every compiled ext
from mmdet3d.ops import Voxelization
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d

dev = 'cuda'
pts = torch.rand(20000, 5, device=dev) * torch.tensor([40, 40, 4, 1, 1], device=dev) - torch.tensor([20, 20, 2, 0, 0], device=dev)
vox = Voxelization([0.2, 0.2, 0.2], [-20, -20, -2, 20, 20, 2], -1, -1).to(dev)
coors = vox(pts)
print('dynamic voxelize', coors.shape, (coors >= 0).all().item())

boxes = torch.tensor([[0, 0, -1, 4, 2, 1.5, 0.3], [5, 5, -1, 4, 2, 1.5, 0.0]], device=dev)
ids = points_in_boxes_gpu(pts[None, :, :3].contiguous(), boxes[None])
n_in = [(ids == i).sum().item() for i in range(2)]
lb = LiDARInstance3DBoxes(boxes)
cpu_n = [lb[i:i+1].points_in_boxes(pts[:, :3]).ne(-1).sum().item() for i in range(2)]
print('points_in_boxes gpu', n_in, 'recheck', cpu_n)
iou = bbox_overlaps_3d(boxes, boxes, coordinate='lidar')
print('iou3d', iou)
assert torch.allclose(iou.diag(), torch.ones(2, device=dev), atol=1e-4)
print('aligned_iou_3d', LiDARInstance3DBoxes.aligned_iou_3d(lb, lb))

from mmcv import Config
from mmdet3d.models import build_model
cfg = Config.fromfile(sys.argv[1] if len(sys.argv) > 1 else 'configs/ctrl/ctrl_veh_24e.py')
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).to(dev)
print('CTRL model', type(model).__name__, sum(p.numel() for p in model.parameters()) / 1e6, 'M params')
