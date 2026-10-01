"""Verify a CTRL checkpoint really lands in the model (spconv2 weight layouts are easy to get silently wrong).

1) load the given checkpoint, compare every *live* parameter (module._parameters, not state_dict()) to the file;
2) round-trip: mmcv save_checkpoint -> load into a fresh model -> live params must be identical.
"""
import sys
import tempfile
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet3d.models import build_model


def live_params(model):
    return {f'{mn}.{pn}' if mn else pn: p.detach() for mn, m in model.named_modules()
            for pn, p in m._parameters.items() if p is not None}


def build(cfg):
    return build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))


cfg = Config.fromfile(sys.argv[1])
ckpt = torch.load(sys.argv[2], map_location='cpu', weights_only=False)['state_dict']
model = build(cfg)
load_checkpoint(model, sys.argv[2], map_location='cpu', strict=False)
bad = []
for k, p in live_params(model).items():
    w = ckpt[k]
    if w.shape != p.shape:
        w = w.permute(*([w.dim() - 1] + list(range(w.dim() - 1))))  # (D,H,W,in,out) -> (out,D,H,W,in)
    if w.shape != p.shape or not torch.equal(w, p):
        bad.append(k)
print(f'[load official] params={len(ckpt)} not_loaded={len(bad)} {bad[:3]}')

with tempfile.NamedTemporaryFile(suffix='.pth') as f:
    save_checkpoint(model, f.name)
    model2 = build(cfg)
    load_checkpoint(model2, f.name, map_location='cpu', strict=False)
    a, b = live_params(model), live_params(model2)
    diff = [k for k in a if not torch.equal(a[k], b[k])]
print(f'[save->load roundtrip] differing params={len(diff)} {diff[:3]}')
