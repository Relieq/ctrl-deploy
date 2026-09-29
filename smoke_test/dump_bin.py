"""Print a few objects and field statistics of Waymo .bin files."""
import sys
import numpy as np
from waymo_open_dataset.protos import metrics_pb2

for path in sys.argv[1:]:
    objs = metrics_pb2.Objects()
    objs.ParseFromString(open(path, 'rb').read())
    o = objs.objects
    types = np.array([x.object.type for x in o])
    scores = np.array([x.score for x in o])
    lwh = np.array([[x.object.box.length, x.object.box.width, x.object.box.height] for x in o])
    print(path.split('/')[-1], 'n', len(o), 'types', dict(zip(*np.unique(types, return_counts=True))),
          'score[min/mean/max]', scores.min().round(3), scores.mean().round(3), scores.max().round(3),
          'lwh mean', lwh.mean(0).round(2), 'cz mean', np.mean([x.object.box.center_z for x in o]).round(2))
    print('  sample:', str(o[0]).replace('\n', ' ')[:300])
