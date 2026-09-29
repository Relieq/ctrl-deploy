"""Runtime fixes for SST/tools/ctrl and ImmortalTracker-for-CTRL (single GPU, PyYAML>=6, partial Waymo subsets).

Usage: python tools_patch.py <CTRL_ROOT>. Idempotent.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
SST = root / 'SST'
TRK = root / 'ImmortalTracker-for-CTRL'


def sub(path, pairs, regex=False):
    p = Path(path)
    s0 = s = p.read_text()
    for a, b in pairs:
        s = re.sub(a, b, s) if regex else s.replace(a, b)
    if s != s0:
        p.write_text(s)
        print('patched', p)


# PyYAML >= 6 requires an explicit Loader
for f in [TRK / 'main_waymo.py', TRK / 'main_nuscenes.py', TRK / 'evaluation/waymo/pred_bin.py',
          TRK / 'evaluation/waymo/rm_fp_by_gt.py', SST / 'tools/ctrl/extend_tracks.py',
          SST / 'tools/ctrl/generate_track_input.py', SST / 'tools/ctrl/generate_candidates.py']:
    sub(f, [(r"yaml\.load\((open\([^)]*\))\)", r"yaml.load(\1, Loader=yaml.FullLoader)")], regex=True)

# Workers assumed 8 GPUs: map token onto the GPUs actually present
for f in [TRK / 'main_waymo.py', TRK / 'evaluation/waymo/rm_fp_by_gt.py',
          SST / 'tools/ctrl/generate_track_input.py', SST / 'tools/ctrl/generate_candidates.py',
          SST / 'tools/ctrl/remove_empty.py']:
    sub(f, [('torch.cuda.set_device(token % 8)', 'torch.cuda.set_device(token % torch.cuda.device_count())')])

# numpy >= 1.24 removed np.int
for f in [TRK / 'mot_3d/preprocessing/bbox_coarse_hash.py', TRK / 'preprocessing/py_nms/bbox_coarse_hash.py']:
    sub(f, [(r'\bnp\.int\b(?!\d|_)', 'np.int64')], regex=True)

# Single-process code paths called main()/process_single_frame() with wrong arguments
sub(TRK / 'preparedata/waymo/ego_info.py',
    [('        main(args.data_folder, args.output_folder)', '        main(0, 1, args.data_folder, args.output_folder)')])
sub(SST / 'tools/ctrl/remove_empty.py',
    [('process_single_frame(config, waymo_data_root, out_list, input_list, args.split, ts2idx, 0, 1)',
      'process_single_frame(config, waymo_data_root, out_list, input_list, ts2idx, args.split, 0, 1)')])

# protobuf >= 4 (upb) rejects bytearray in ParseFromString
for f in [SST / 'tools/ctrl/generate_train_gt_bin.py', SST / 'tools/data_converter/waymo_converter.py',
          SST / 'tools/test_waymo.py', SST / 'mmdet3d/core/evaluation/waymo_utils/prediction_kitti_to_waymo.py',
          TRK / 'preparedata/waymo/time_stamp.py', TRK / 'preparedata/waymo/ego_info.py']:
    sub(f, [(r'ParseFromString\(bytearray\(', 'ParseFromString(bytes(')], regex=True)

# torch>=2 launchers pass --local-rank (dash) instead of --local_rank
for f in [SST / 'tools/train.py', SST / 'tools/test.py']:
    sub(f, [("parser.add_argument('--local_rank', type=int", "parser.add_argument('--local_rank', '--local-rank', type=int")])

# mmdet >= 2.25 train_detector() reads cfg.device
sub(SST / 'tools/train.py', [("    cfg.seed = args.seed\n",
                              "    cfg.seed = args.seed\n    cfg.device = cfg.get('device', 'cuda')\n")])

# extract_poses.py: allow a subset of splits (e.g. only validation downloaded)
sub(SST / 'tools/ctrl/extract_poses.py', [
    ('    for path in info_path_list:\n        with open(path',
     '    for path in info_path_list:\n        if not os.path.exists(path):\n'
     '            print(f\'skip missing {path}\')\n            continue\n        with open(path'),
    ('    assert sum([len(v) for _, v in context2ts.items()]) == len(idx2timestamp)\n'
     '    assert len(pose_dict) == len(idx2timestamp)',
     '    assert sum([len(v) for _, v in context2ts.items()]) == len(pose_dict)'),
])
