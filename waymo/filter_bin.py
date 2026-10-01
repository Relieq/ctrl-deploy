"""Keep only objects whose context_name is among the converted segments (gt_parts/*.bin)."""
import argparse
import glob
import os.path as osp

from waymo_open_dataset.protos import metrics_pb2

parser = argparse.ArgumentParser()
parser.add_argument('src')
parser.add_argument('dst')
parser.add_argument('--gt-parts', required=True)
args = parser.parse_args()

contexts = {osp.basename(p)[:-4] for p in glob.glob(osp.join(args.gt_parts, '*.bin'))}
src = metrics_pb2.Objects()
src.ParseFromString(open(args.src, 'rb').read())
dst = metrics_pb2.Objects()
dst.objects.extend(o for o in src.objects if o.context_name in contexts)
open(args.dst, 'wb').write(dst.SerializeToString())
print(f'{len(dst.objects)}/{len(src.objects)} objects kept ({len(contexts)} segments)')
