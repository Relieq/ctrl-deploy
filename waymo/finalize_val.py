"""After stream_val.sh: assemble the lookup tables, gt.bin and subset-filtered detection bins for CTRL on Waymo val.

Writes under <sst_data>/waymo:
  waymo_format/gt.bin                               (merged gt_parts; only converted segments)
  kitti_format/{idx2timestamp,idx2contextname}.pkl  (restricted to converted val frames)
  kitti_format/poses.pkl                            (authors' poses.pkl restricted to those frames)
  kitti_format/context2timestamp.pkl
and <sst_data>/ctrl_bins/validation/<name>.bin for every --det bin, keeping only converted segments.
"""
import argparse
import glob
import os
import os.path as osp
import pickle as pkl
from collections import defaultdict

from waymo_open_dataset.protos import metrics_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--sst', required=True)
parser.add_argument('--sst-data', required=True)
parser.add_argument('--poses', required=True, help="authors' poses.pkl {timestamp: 4x4}")
parser.add_argument('--det', nargs='*', default=[], help='detection bins to filter to the converted segments')
args = parser.parse_args()

wfmt = osp.join(args.sst_data, 'waymo/waymo_format')
kitti = osp.join(args.sst_data, 'waymo/kitti_format')
parts = sorted(glob.glob(osp.join(wfmt, 'gt_parts/*.bin')))
contexts = {osp.basename(p)[:-4] for p in parts}
assert contexts, 'no converted segments found'

gt = metrics_pb2.Objects()
for p in parts:
    o = metrics_pb2.Objects()
    o.ParseFromString(open(p, 'rb').read())
    gt.objects.extend(o.objects)
open(osp.join(wfmt, 'gt.bin'), 'wb').write(gt.SerializeToString())

idx2ts = pkl.load(open(osp.join(args.sst, 'tools/idx2timestamp.pkl'), 'rb'))
idx2ctx = pkl.load(open(osp.join(args.sst, 'tools/idx2contextname.pkl'), 'rb'))
keep = [i for i, c in idx2ctx.items() if i[0] == '1' and c in contexts]
velo = osp.join(kitti, 'training/velodyne')
missing = [i for i in keep if not osp.exists(osp.join(velo, i + '.bin'))]
assert not missing, f'{len(missing)} velodyne files missing, e.g. {missing[:3]}'

poses_all = pkl.load(open(args.poses, 'rb'))
pkl.dump({i: idx2ts[i] for i in keep}, open(osp.join(kitti, 'idx2timestamp.pkl'), 'wb'))
pkl.dump({i: idx2ctx[i] for i in keep}, open(osp.join(kitti, 'idx2contextname.pkl'), 'wb'))
pkl.dump({idx2ts[i]: poses_all[idx2ts[i]] for i in keep}, open(osp.join(kitti, 'poses.pkl'), 'wb'))
c2ts = defaultdict(list)
for i in keep:
    c2ts[idx2ctx[i]].append(idx2ts[i])
pkl.dump({c: sorted(v) for c, v in c2ts.items()}, open(osp.join(kitti, 'context2timestamp.pkl'), 'wb'))

out_dir = osp.join(args.sst_data, 'ctrl_bins/validation')
os.makedirs(out_dir, exist_ok=True)
for path in args.det:
    src = metrics_pb2.Objects()
    src.ParseFromString(open(path, 'rb').read())
    dst = metrics_pb2.Objects()
    dst.objects.extend(o for o in src.objects if o.context_name in contexts)
    out = osp.join(out_dir, osp.basename(path))
    open(out, 'wb').write(dst.SerializeToString())
    print(f'{osp.basename(path)}: {len(dst.objects)}/{len(src.objects)} objects kept -> {out}')

print(f'segments={len(contexts)} frames={len(keep)} gt_objects={len(gt.objects)}')
