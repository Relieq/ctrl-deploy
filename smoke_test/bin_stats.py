"""Compare prediction .bin files against gt.bin: #objects, #track ids, recall/precision by BEV center distance < 1 m."""
import sys
from collections import defaultdict
import numpy as np
from waymo_open_dataset.protos import metrics_pb2


def load(path):
    objs = metrics_pb2.Objects()
    objs.ParseFromString(open(path, 'rb').read())
    per_frame = defaultdict(list)
    ids = set()
    for o in objs.objects:
        b = o.object.box
        per_frame[(o.context_name, o.frame_timestamp_micros)].append((b.center_x, b.center_y, b.heading, o.score))
        ids.add(o.object.id)
    return per_frame, len(objs.objects), len(ids)


gt, n_gt, n_gt_ids = load(sys.argv[1])
print(f'{"gt":<40} boxes={n_gt:5d} ids={n_gt_ids}')
for path in sys.argv[2:]:
    pred, n, n_ids = load(path)
    tp = 0
    for k, g in gt.items():
        p = np.array(pred.get(k, [])).reshape(-1, 4)
        for gx, gy, *_ in g:
            if len(p) and np.min(np.hypot(p[:, 0] - gx, p[:, 1] - gy)) < 1.0:
                tp += 1
    print(f'{path.split("/")[-1]:<40} boxes={n:5d} ids={n_ids:4d} recall={tp / n_gt:.3f} precision~={tp / max(n, 1):.3f}')
