"""Convert ONE Waymo validation tfrecord into everything the CTRL + ImmortalTracker pipeline needs.

Unlike SST's create_data.py, the segment index is looked up from idx2contextname.pkl (the table CTRL and the
authors' poses.pkl are keyed by), so any subset of segments can be converted in any order.

Writes:
  <sst_data>/waymo/kitti_format/training/velodyne/1SSSFFF.bin   (N x 6 float32: x y z intensity elongation ts; SST converter)
  <sst_data>/waymo/waymo_format/gt_parts/<ctx>.bin             (laser_labels as metrics_pb2, merged later into gt.bin)
  <trk_data>/waymo/validation/ts_info/segment-<ctx>_with_camera_labels.json
  <trk_data>/waymo/validation/ego_info/segment-<ctx>_with_camera_labels.npz
"""
import argparse
import json
import os
import os.path as osp
import pickle as pkl
import sys

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import metrics_pb2

parser = argparse.ArgumentParser()
parser.add_argument('tfrecord')
parser.add_argument('--sst', required=True, help='SST repo root (converter code + tools/idx2contextname.pkl)')
parser.add_argument('--sst-data', required=True, help='directory that holds waymo/{kitti_format,waymo_format}')
parser.add_argument('--trk-data', required=True, help='directory that holds waymo/validation/{ts_info,ego_info}')
args = parser.parse_args()

sys.path.insert(0, args.sst)
from tools.data_converter.waymo_converter import Waymo2KITTI  # noqa: E402

idx2ctx = pkl.load(open(osp.join(args.sst, 'tools/idx2contextname.pkl'), 'rb'))
ctx2seg = {ctx: int(idx[1:4]) for idx, ctx in idx2ctx.items() if idx[0] == '1'}  # prefix 1 = validation

kitti = osp.join(args.sst_data, 'waymo/kitti_format/training')
gt_dir = osp.join(args.sst_data, 'waymo/waymo_format/gt_parts')
ts_dir = osp.join(args.trk_data, 'waymo/validation/ts_info')
ego_dir = osp.join(args.trk_data, 'waymo/validation/ego_info')
for d in (gt_dir, ts_dir, ego_dir):
    os.makedirs(d, exist_ok=True)

# Reuse SST's lidar conversion (range image -> 6-dim points); bypass its folder scan and other outputs.
conv = Waymo2KITTI.__new__(Waymo2KITTI)
conv.prefix = '1'
conv.point_cloud_save_dir = osp.join(kitti, 'velodyne')
os.makedirs(conv.point_cloud_save_dir, exist_ok=True)
conv.filter_no_label_zone_points = True
conv.lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']

gt = metrics_pb2.Objects()
ts_list, ego = [], {}
ctx = None
for frame_idx, data in enumerate(tf.data.TFRecordDataset(args.tfrecord, compression_type='')):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytes(data.numpy()))
    if ctx is None:
        ctx = frame.context.name
        if ctx not in ctx2seg:
            sys.exit(f'{ctx} is not a validation segment in idx2contextname.pkl')
        seg = ctx2seg[ctx]
    expected = f'1{seg:03d}{frame_idx:03d}'
    assert idx2ctx.get(expected) == ctx, f'frame index mismatch at {expected}'
    conv.save_lidar(frame, seg, frame_idx)

    for label in frame.laser_labels:  # same fields as SST tools/ctrl/generate_train_gt_bin.py
        o = gt.objects.add()
        o.context_name = ctx
        o.frame_timestamp_micros = frame.timestamp_micros
        o.object.box.CopyFrom(label.box)
        o.object.id = label.id
        o.object.type = label.type
        o.object.metadata.CopyFrom(label.metadata)
        o.object.num_lidar_points_in_box = label.num_lidar_points_in_box
        o.object.detection_difficulty_level = label.detection_difficulty_level
        o.object.tracking_difficulty_level = label.tracking_difficulty_level

    ts_list.append(frame.timestamp_micros)
    ego[str(frame_idx)] = np.reshape(np.array(frame.pose.transform), [4, 4])

name = f'segment-{ctx}_with_camera_labels'
json.dump(ts_list, open(osp.join(ts_dir, name + '.json'), 'w'))
np.savez_compressed(osp.join(ego_dir, name + '.npz'), **ego)
open(osp.join(gt_dir, ctx + '.bin'), 'wb').write(gt.SerializeToString())
print(f'OK seg={seg:03d} frames={len(ts_list)} gt={len(gt.objects)} {ctx}')
