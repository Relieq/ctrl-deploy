"""Smoke test: run torch_scatter, spconv2 and TorchEx kernels on the GPU."""
import torch
import torch_scatter
import spconv.pytorch as spconv

dev = 'cuda'
src = torch.rand(1000, 16, device=dev)
idx = torch.randint(0, 50, (1000,), device=dev)
print('torch_scatter', torch_scatter.__version__, torch_scatter.scatter_max(src, idx, dim=0)[0].shape)

coors = torch.unique(torch.randint(0, 40, (2000, 3), device=dev), dim=0)
coors = torch.cat([torch.zeros_like(coors[:, :1]), coors], 1).int()
feats = torch.rand(len(coors), 16, device=dev)
x = spconv.SparseConvTensor(feats, coors, [41, 41, 41], 1)
net = torch.nn.Sequential(
    spconv.SubMConv3d(16, 32, 3, indice_key='s1'),
    spconv.SparseConv3d(32, 64, 3, stride=2),
).to(dev)
y = net(x)
print('spconv', spconv.__version__ if hasattr(spconv, '__version__') else '', y.features.shape, y.features.abs().sum().item() > 0)

import ingroup_indices, dynamic_point_pool_ext  # noqa: F401
from torchex import boxes_overlap_1to1
# BEV boxes (N, 5): [x1, y1, x2, y2, ry]; identical 2x4 boxes -> overlap 8
b = torch.tensor([[0, 0, 2, 4, 0.3]], device=dev).repeat(4, 1)
ov = boxes_overlap_1to1(b, b)
print('torchex boxes_overlap_1to1', ov)
assert torch.allclose(ov, torch.full_like(ov, 8.0), atol=1e-3)
