"""Port SST (mmdet3d 0.15 fork) to torch 2.x / mmcv 1.7 / numpy 1.23. Idempotent; run from SST root."""
import re
from pathlib import Path

root = Path('.')

# 1) setup.py: drop the legacy spconv1 extension (CTRL/FSD use spconv2 from pip)
p = root / 'setup.py'
s = p.read_text()
start = s.find("            make_cuda_ext(\n                name='sparse_conv_ext'")
if start != -1:
    end = s.find("            make_cuda_ext(\n                name='iou3d_cuda'", start)
    s = s[:start] + s[end:]
    p.write_text(s)

# 2) THC was removed in torch 1.11: strip includes/extern state, use data_ptr / is_cuda
for f in (root / 'mmdet3d' / 'ops').rglob('*'):
    if f.suffix not in ('.cpp', '.cu', '.h', '.cuh') or 'spconv' in f.parts:
        continue
    s0 = s = f.read_text()
    s = re.sub(r'#include <THC/THC\.h>\n', '#include <ATen/cuda/CUDAContext.h>\n', s)
    s = re.sub(r'extern THCState \*state;\n', '', s)
    s = re.sub(r'\.data<', '.data_ptr<', s)
    s = s.replace('.type().is_cuda()', '.is_cuda()')
    s = s.replace('AT_CHECK(', 'TORCH_CHECK(')
    if s != s0:
        f.write_text(s)
        print('patched', f)

# 4) Python-level API renames (numba >= 0.49, numpy >= 1.20)
PY_SUBS = [
    (r'from numba\.errors import', 'from numba.core.errors import'),
    (r'\bnp\.int\b(?!\d|_)', 'np.int64'),
    (r'\bnp\.float\b(?!\d|_)', 'np.float64'),
    (r'\bnp\.bool\b(?!_)', 'np.bool_'),
    (r'\bnp\.long\b', 'np.int64'),
    (r'\bnp\.object\b(?!_)', 'object'),
]
for f in list((root / 'mmdet3d').rglob('*.py')) + list((root / 'tools').rglob('*.py')):
    s0 = s = f.read_text()
    for a, b in PY_SUBS:
        s = re.sub(a, b, s)
    if s != s0:
        f.write_text(s)
        print('patched', f)

# 5) SIRLayer mutated the config list in place (rel_mlp grows by one layer each time a model is built from the same cfg)
p = root / 'mmdet3d' / 'models' / 'voxel_encoders' / 'voxel_encoder.py'
s = p.read_text()
s = s.replace("            rel_mlp_hidden_dims.append(in_channels) # not self.in_channels\n",
              "            rel_mlp_hidden_dims = list(rel_mlp_hidden_dims) + [in_channels]  # not self.in_channels; do not mutate cfg\n")
p.write_text(s)

# 3) mmdet3d version gate: allow mmcv 1.7.x
p = root / 'mmdet3d' / '__init__.py'
s = p.read_text().replace("mmcv_maximum_version = '1.4.0'", "mmcv_maximum_version = '1.7.2'")
p.write_text(s)
