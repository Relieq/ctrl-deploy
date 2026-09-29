# ctrl-deploy

Môi trường triển khai cho pipeline auto-labeling **CTRL** ([tusen-ai/SST](https://github.com/tusen-ai/SST)) kết hợp
**ImmortalTracker-for-CTRL** ([Abyssaledge/ImmortalTracker-for-CTRL](https://github.com/Abyssaledge/ImmortalTracker-for-CTRL))
trên GPU Blackwell (RTX 50xx, sm_120), chạy trong WSL2 Ubuntu.

Bộ gốc của SST là PyTorch 1.8 + CUDA 11.1, không chạy được trên sm_120. Repo này port toàn bộ stack lên
PyTorch 2.7 + CUDA 12.8, dựng lại mọi CUDA extension, vá các lỗi tương thích và cung cấp smoke test end-to-end.

## Stack đã kiểm chứng

| Thành phần | Phiên bản gốc | Phiên bản dùng ở đây |
|---|---|---|
| Python | 3.8 | 3.10 |
| PyTorch / CUDA | 1.8.0 / 11.1 | 2.7.1 / 12.8 (nvcc 12.8, gcc 12 từ conda) |
| mmcv-full | 1.3.9 | 1.7.2 (build từ source, C++17, vá `_get_stream`) |
| mmdet / mmseg | 2.14.0 / 0.14.1 | 2.28.2 / 0.30.0 |
| mmdet3d (fork trong SST) | 0.15.0 | 0.15.0 (vá THC, numba, numpy) |
| spconv | 2.2.3 (cu113) | spconv-cu126 2.3.8 |
| torch_scatter | 2.0.9 | 2.1.2 (pt27cu128) |
| TorchEx | - | build từ source cho sm_120 |
| TensorFlow / WOD | 2.4 / tf-2-4-0 1.4.1 | tensorflow-cpu 2.12 / tf-2-12-0 1.6.4 (`--no-deps`) |
| numpy / numba | <1.20 / 0.48 | 1.23.5 / 0.58.1 |
| Waymo metric tools | - | build bằng Bazel từ WOD v1.6.1 |

## Cách hai repo kết hợp

```
FSD detections (.bin, chuẩn Waymo)
  └─ ImmortalTracker: preparedata (ts_info, ego_info) → detection.py → main_waymo.py → pred_bin.py
       └─ tracks .bin (box ego từng frame, id "<type>_<id>", keep10 = giữ tối đa 10 frame dự đoán Kalman)
            └─ SST tools/ctrl: [extend_tracks.py: ngoại suy ngược] → generate_track_input.py
                 → [train] generate_candidates.py → tools/train.py configs/ctrl/*.py
                 → [infer] tools/test.py → .bin đã tinh chỉnh → [remove_empty.py]
```

Hai repo chỉ nối với nhau qua file `.bin` (protobuf `metrics_pb2.Objects`). Chúng dùng chung một conda env
`ctrl`, vì tracker cần `mmdet3d<1.0` (lấy từ SST) để tính IoU 3D trên GPU, và cả hai import `waymo_open_dataset`
ngay khi nạp module.

## Cài đặt

Yêu cầu: Windows 11 + WSL2 Ubuntu, driver NVIDIA hỗ trợ CUDA ≥ 12.8.

1. Cấp đủ RAM cho WSL. Chép `wslconfig.example` thành `%UserProfile%\.wslconfig`, rồi chạy `wsl --shutdown`.
   Nên đặt distro ở ổ nhiều dung lượng, ví dụ `wsl --manage Ubuntu --move D:\WSL\Ubuntu`.
2. Trong WSL:
   ```bash
   bash setup_all.sh              # khoảng 30-60 phút; mã nguồn nằm tại $CTRL_ROOT (mặc định ~/CTRL)
   source scripts/env.sh          # kích hoạt env cho các phiên sau
   bash smoke_test/run_all.sh     # kiểm tra end-to-end trên dữ liệu giả lập
   ```

Các script đều idempotent và có thể chạy lại từng bước (`scripts/0X_*.sh`).

| Script | Việc làm |
|---|---|
| `01_miniforge.sh` | Cài Miniforge và build-essential |
| `02_env.sh` | Tạo env `ctrl`: python 3.10, nvcc 12.8, gcc 12, torch 2.7.1+cu128 |
| `03_clone.sh` | Clone SST, ImmortalTracker-for-CTRL, TorchEx (ghim commit) và mmcv v1.7.2 |
| `04_mmcv.sh` | Build mmcv-full 1.7.2 cho sm_120 |
| `05_waymo_tf.sh` | Cài tensorflow-cpu 2.12 + waymo-open-dataset (chỉ dùng protos/utils) |
| `06_openmmlab_sparse.sh` | Cài mmdet, mmseg, torch_scatter, spconv; build TorchEx |
| `07_sst.sh` | Port SST sang torch 2 (`sst_port_torch2.py`), vá tool (`tools_patch.py`), build ops |
| `08_waymo_eval_tools.sh` | Build `compute_detection_metrics_main` / `compute_tracking_metrics_main` |

## Các bản vá (tóm tắt)

- **mmcv 1.7.2**: `-std=c++17`; `_get_stream(torch.device(...))` để chạy với torch ≥ 2.
- **SST ops**: bỏ `THC` (đã bị xoá khỏi torch 1.11) và dùng `ATen/cuda/CUDAContext.h`; đổi `.data<T>()` thành
  `.data_ptr<T>()`; bỏ extension `sparse_conv_ext` (spconv1 cũ, CTRL/FSD không dùng).
- **SST Python**: `numba.errors` đổi thành `numba.core.errors`; `np.int/np.float/np.bool` đổi thành kiểu cụ thể;
  nới giới hạn phiên bản mmcv; `cfg.device` cho mmdet 2.28; `--local-rank` cho launcher của torch 2.
- **tools/ctrl và ImmortalTracker**:
  - `yaml.load(..., Loader=FullLoader)`.
  - `set_device(token % device_count())`: bản gốc giả định có 8 GPU nên lỗi âm thầm khi chỉ có 1 GPU.
  - `ParseFromString(bytes(...))` cho protobuf 4.
  - Sửa đường chạy một tiến trình của `ego_info.py` và `remove_empty.py`.
  - `extract_poses.py` chấp nhận thiếu split.

## Smoke test

`smoke_test/make_synthetic.py` sinh 2 segment × 60 frame. Dữ liệu dùng context name và timestamp thật của tập
validation Waymo, xe chuyển động, point cloud, detection có nhiễu và bỏ sót 20%, cùng tfrecord `Frame` thật.
Script `run_all.sh` chạy toàn bộ pipeline: preparedata → tracking → extend → track input → candidates →
train 1 epoch → inference → eval.

Kết quả tham khảo trên RTX 5050 Laptop 8 GB:

| Bước | Kết quả |
|---|---|
| Detection giả lập | recall 0.78 (bỏ sót 20%) |
| ImmortalTracker (keep10) | recall 0.996, Vehicle L2 MOTA 0.869 |
| extend_tracks (backward) | Vehicle L1 mAP 0.940 |
| CTRL, 1 epoch (`run_train_test.sh`) | mAP ≈ 0: chưa học đủ, chỉ để kiểm tra luồng chạy |
| CTRL, 40 epoch (`run_learn_check.sh`) | Vehicle L1 mAP 0.941 / mAPH 0.932 |

Train CTRL batch 2 dùng khoảng 2.6 GB VRAM.

## Chạy với dữ liệu Waymo thật

1. Tải Waymo Open Dataset v1.4.x (cần đăng ký tại waymo.com/open) vào `SST/data/waymo/waymo_format/{training,validation,testing}`,
   rồi chuyển sang định dạng KITTI theo `SST/docs/overall_instructions.md`:
   `python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 8 --extra-tag waymo`.
   Sau đó chép `tools/idx2timestamp.pkl` và `tools/idx2contextname.pkl` vào `data/waymo/kitti_format/`.
2. `python tools/ctrl/extract_poses.py` (và `generate_train_gt_bin.py` nếu cần train).
3. Tracker: trong `ImmortalTracker-for-CTRL`, chạy như `smoke_test/run_tracker.sh` với file detection FSD thật
   (`.bin`). Khi sinh dữ liệu train, bỏ khối `merge:` trong config tracker.
4. CTRL: sửa `bin_path`/`val_bin_path`/`split` trong `tools/ctrl/data_configs/fsd_base_vehicle.yaml`,
   chạy `generate_track_input.py` (và `generate_candidates.py` khi train), rồi chạy
   `python tools/test.py configs/ctrl/ctrl_veh_24e.py <ckpt> --eval waymo --options pklfile_prefix=./work_dirs/ctrl_val`.

Lưu ý với GPU 8 GB: inference chạy thoải mái. Train cấu hình gốc (8 GPU × 16 mẫu) cần giảm `samples_per_gpu`
và chạy lâu hơn nhiều.
