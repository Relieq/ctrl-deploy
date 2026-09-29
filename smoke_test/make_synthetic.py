"""Generate a tiny synthetic Waymo-like dataset that exercises the whole CTRL pipeline.

Uses real validation context names / timestamps from SST/tools/idx2*.pkl so every
lookup table (idx2timestamp, ts2idx, context2timestamp) behaves like the real thing.

Outputs (under --out):
  sst/data/waymo/waymo_format/validation/segment-<ctx>_with_camera_labels.tfrecord  (Frame protos: pose, ts, laser_labels)
  sst/data/waymo/waymo_format/gt.bin                                                (metrics_pb2 GT)
  sst/data/waymo/kitti_format/{idx2timestamp,idx2contextname,waymo_infos_val}.pkl
  sst/data/waymo/kitti_format/training/velodyne/<idx>.bin                           (N x 6 float32, vehicle frame)
  det/fsd_synth_val.bin                                                             (noisy detections, metrics_pb2)
"""
import argparse
import os
import os.path as osp
import pickle as pkl

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import metrics_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--sst', required=True, help='SST repo root (for tools/idx2*.pkl)')
parser.add_argument('--out', required=True)
parser.add_argument('--segments', type=int, default=2)
parser.add_argument('--frames', type=int, default=60)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

idx2ts = pkl.load(open(osp.join(args.sst, 'tools/idx2timestamp.pkl'), 'rb'))
idx2ctx = pkl.load(open(osp.join(args.sst, 'tools/idx2contextname.pkl'), 'rb'))

sst_data = osp.join(args.out, 'sst/data/waymo')
kitti = osp.join(sst_data, 'kitti_format')
wfmt = osp.join(sst_data, 'waymo_format')
velo_dir = osp.join(kitti, 'training/velodyne')
for d in (velo_dir, osp.join(wfmt, 'validation'), osp.join(args.out, 'det')):
    os.makedirs(d, exist_ok=True)


def pose_at(t, yaw_rate):
    """Ego drives forward at 6 m/s while turning slowly."""
    yaw = yaw_rate * t
    x = 6.0 * np.sin(yaw) / yaw_rate if yaw_rate else 6.0 * t
    y = 6.0 * (1 - np.cos(yaw)) / yaw_rate if yaw_rate else 0.0
    T = np.eye(4)
    T[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    T[:3, 3] = [x, y, 0.0]
    return T


def box_points(c, lwh, yaw, n):
    """Points on the visible shell of a box (vehicle-frame), plus intensity/elongation/0."""
    l, w, h = lwh
    local = rng.uniform(-0.5, 0.5, (n, 3)) * [l, w, h]
    face = rng.integers(0, 3, n)
    local[np.arange(n), face] = np.sign(local[np.arange(n), face]) * np.array([l, w, h])[face] / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = local[:, :2] @ R.T + c[:2]
    z = local[:, 2] + c[2]
    feats = np.stack([rng.uniform(0, 1, n), rng.uniform(0, 0.3, n), np.zeros(n)], 1)
    return np.concatenate([xy, z[:, None], feats], 1)


gt_objs, det_objs, infos = metrics_pb2.Objects(), metrics_pb2.Objects(), []
for s in range(args.segments):
    prefix = f'1{s:03d}'
    idxs = [f'{prefix}{f:03d}' for f in range(args.frames)]
    ctx = idx2ctx[idxs[0]]
    tss = [idx2ts[i] for i in idxs]
    t0 = tss[0]
    yaw_rate = rng.uniform(-0.05, 0.05)

    # cars in world frame: start position relative to ego start, constant velocity
    n_cars = 6
    cars = []
    for k in range(n_cars):
        heading = rng.uniform(-np.pi, np.pi) if k == 0 else rng.choice([0.0, np.pi]) + rng.normal(0, 0.1)
        speed = 0.0 if k == 0 else rng.uniform(2, 10)
        cars.append(dict(
            id=f'synth_{s}_{k}',
            p0=np.array([rng.uniform(-10, 50), rng.uniform(-15, 15), 0.0]),
            v=speed * np.array([np.cos(heading), np.sin(heading), 0.0]),
            heading=heading,
            lwh=np.array([rng.uniform(4, 5), rng.uniform(1.8, 2.1), rng.uniform(1.5, 1.8)]),
        ))

    writer = tf.io.TFRecordWriter(osp.join(wfmt, 'validation', f'segment-{ctx}_with_camera_labels.tfrecord'))
    for f, (idx, ts) in enumerate(zip(idxs, tss)):
        t = (ts - t0) * 1e-6
        T = pose_at(t, yaw_rate)
        Tinv = np.linalg.inv(T)
        ego_yaw = np.arctan2(T[1, 0], T[0, 0])

        frame = dataset_pb2.Frame()
        frame.context.name = ctx
        frame.timestamp_micros = ts
        frame.pose.transform.extend(T.flatten().tolist())

        pts = [np.concatenate([rng.uniform([-60, -60, -0.05], [60, 60, 0.05], (4000, 3)),
                               rng.uniform(0, 1, (4000, 3)) * [1, 0.3, 0]], 1)]
        for car in cars:
            cw = car['p0'] + car['v'] * t
            cw[2] = car['lwh'][2] / 2
            cv = (Tinv @ np.append(cw, 1))[:3]
            yaw_v = car['heading'] - ego_yaw
            dist = np.linalg.norm(cv[:2])
            if dist > 70:
                continue
            n_pts = int(np.clip(4000 / (dist + 5), 20, 600))
            pts.append(box_points(cv, car['lwh'], yaw_v, n_pts))

            box = label_pb2.Label.Box(center_x=cv[0], center_y=cv[1], center_z=cv[2],
                                      length=car['lwh'][0], width=car['lwh'][1], height=car['lwh'][2],
                                      heading=float(np.arctan2(np.sin(yaw_v), np.cos(yaw_v))))
            lab = frame.laser_labels.add()
            lab.box.CopyFrom(box)
            lab.type = label_pb2.Label.TYPE_VEHICLE
            lab.id = car['id']
            lab.num_lidar_points_in_box = n_pts

            o = gt_objs.objects.add()
            o.object.box.CopyFrom(box)
            o.object.type = label_pb2.Label.TYPE_VEHICLE
            o.object.id = car['id']
            o.object.num_lidar_points_in_box = n_pts
            o.context_name = ctx
            o.frame_timestamp_micros = ts

            if rng.uniform() < 0.8:  # detector misses ~20% of frames
                d = det_objs.objects.add()
                d.object.box.CopyFrom(box)
                d.object.box.center_x += rng.normal(0, 0.15)
                d.object.box.center_y += rng.normal(0, 0.15)
                d.object.box.length *= rng.uniform(0.92, 1.08)
                d.object.box.width *= rng.uniform(0.92, 1.08)
                d.object.box.heading += rng.normal(0, 0.05)
                d.object.type = label_pb2.Label.TYPE_VEHICLE
                d.score = float(rng.uniform(0.3, 0.95))
                d.context_name = ctx
                d.frame_timestamp_micros = ts

        if rng.uniform() < 0.3:  # sporadic false positive
            d = det_objs.objects.add()
            d.object.box.CopyFrom(label_pb2.Label.Box(
                center_x=rng.uniform(-40, 40), center_y=rng.uniform(-40, 40), center_z=0.8,
                length=4.5, width=2.0, height=1.6, heading=rng.uniform(-3, 3)))
            d.object.type = label_pb2.Label.TYPE_VEHICLE
            d.score = float(rng.uniform(0.1, 0.4))
            d.context_name = ctx
            d.frame_timestamp_micros = ts

        writer.write(frame.SerializeToString())
        np.concatenate(pts).astype(np.float32).tofile(osp.join(velo_dir, f'{idx}.bin'))
        infos.append({'point_cloud': {'velodyne_path': f'training/velodyne/{idx}.bin', 'num_features': 6},
                      'pose': T, 'timestamp': ts})
    writer.close()

open(osp.join(wfmt, 'gt.bin'), 'wb').write(gt_objs.SerializeToString())
open(osp.join(args.out, 'det/fsd_synth_val.bin'), 'wb').write(det_objs.SerializeToString())
pkl.dump(infos, open(osp.join(kitti, 'waymo_infos_val.pkl'), 'wb'))
# lookup tables restricted to generated frames (generate_track_input iterates every frame listed there)
kept = [i['point_cloud']['velodyne_path'].split('/')[-1][:-4] for i in infos]
pkl.dump({i: idx2ts[i] for i in kept}, open(osp.join(kitti, 'idx2timestamp.pkl'), 'wb'))
pkl.dump({i: idx2ctx[i] for i in kept}, open(osp.join(kitti, 'idx2contextname.pkl'), 'wb'))
print(f'segments={args.segments} frames={len(infos)} gt={len(gt_objs.objects)} dets={len(det_objs.objects)}')
