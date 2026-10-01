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
| TensorFlow / WOD | 2.4 / tf-2-4-0 1.4.1 | tensorflow-cpu 2.12 / tf-2-12-0 1.6.4 (`--no-deps`), protobuf 3.20.3 |
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
| `09_download_ctrl_resources.sh` | Tải checkpoint, detection FSD val và `poses.pkl` công khai của tác giả |
| `10_gcloud.sh` | Cài Google Cloud CLI để tải Waymo |

## Các bản vá (tóm tắt)

- **mmcv 1.7.2**: `-std=c++17`; `_get_stream(torch.device(...))` để chạy với torch ≥ 2.
- **SST ops**: bỏ `THC` (đã bị xoá khỏi torch 1.11) và dùng `ATen/cuda/CUDAContext.h`; đổi `.data<T>()` thành
  `.data_ptr<T>()`; bỏ extension `sparse_conv_ext` (spconv1 cũ, CTRL/FSD không dùng).
- **SST Python**: `numba.errors` đổi thành `numba.core.errors`; `np.int/np.float/np.bool` đổi thành kiểu cụ thể;
  nới giới hạn phiên bản mmcv; `cfg.device` cho mmdet 2.28; `--local-rank` cho launcher của torch 2.
- **tools/ctrl và ImmortalTracker**:
  - `yaml.load(..., Loader=FullLoader)`.
  - `set_device(token % device_count())`: bản gốc giả định có 8 GPU nên lỗi âm thầm khi chỉ có 1 GPU.
  - `ParseFromString(bytes(...))`. Riêng các util của WOD vẫn dùng `bytearray`, nên env ghim `protobuf==3.20.3`.
  - Sửa đường chạy một tiến trình của `ego_info.py` và `remove_empty.py`.
  - `extract_poses.py` chấp nhận thiếu split.
  - Converter Waymo: WOD ≥ 1.5 trả về 4 giá trị từ `parse_range_image_and_camera_projection`.
- **Bug của SST**: `SIRLayer` gọi `append` thẳng vào `rel_mlp_hidden_dims` trong config. Mỗi lần dựng lại model từ
  cùng `cfg`, nó có thêm một lớp và không còn khớp checkpoint. Đã sửa; `scripts/check_ckpt.py` xác nhận cả 3
  checkpoint chính thức nạp đủ 322/322 tham số.

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

## Chạy trên Waymo validation (thư mục `waymo/`)

Yêu cầu: tài khoản Google đã đăng ký tại https://waymo.com/open. Chạy `gcloud auth login` (bằng user dùng để chạy
pipeline, ở đây là `root`) và cài gcloud qua `scripts/10_gcloud.sh`.

```bash
bash scripts/09_download_ctrl_resources.sh        # checkpoint + detection FSD val + poses.pkl công khai của tác giả (~1.7 GB)
NUM_SEGMENTS=20 P=3 bash waymo/stream_val.sh      # tải → convert → xoá tfrecord; bỏ NUM_SEGMENTS để chạy đủ 202
bash waymo/run_val_all.sh                         # vehicle, pedestrian, cyclist; mỗi lớp eval sau từng bước
```

- `waymo/convert_segment.py` lấy chỉ số segment từ `idx2contextname.pkl`, không dựa vào thứ tự file trong thư mục.
  Nhờ vậy có thể convert một tập con, theo thứ tự bất kỳ, và vẫn khớp với `poses.pkl`/`idx2timestamp.pkl` của tác giả.
  Mỗi segment sinh ra velodyne 6 chiều (đúng code converter của SST), GT và `ts_info`/`ego_info` cho tracker,
  chiếm khoảng 0.6 GB. Cả 202 segment khoảng 120 GB.
- `waymo/finalize_val.py` ghép `gt.bin`, các bảng tra cứu và lọc các file `.bin` detection theo các segment đã convert.
- Kết quả của từng bước nằm trong `SST/data/ctrl_runs/<lớp>/summary.txt`.

### Kết quả trên 20 segment val đầu (checkpoint chính thức, RTX 5050 Laptop)

L1 mAP (L2 mAP trong ngoặc):

| Bước | Vehicle | Pedestrian | Cyclist |
|---|---|---|---|
| FSD gốc (`fsd_base_*_val.bin`) | 0.814 (0.746) | 0.864 (0.816) | 0.883 (0.853) |
| ImmortalTracker keep10 | 0.807 (0.740) | 0.872 (0.826) | 0.889 (0.864) |
| + extend ngược | 0.808 (0.741) | 0.872 (0.827) | 0.890 (0.865) |
| CTRL | 0.857 (0.802) | 0.886 (0.843) | 0.914 (0.896) |
| **CTRL + bỏ box rỗng** | **0.862 (0.807)** | **0.893 (0.849)** | **0.917 (0.899)** |
| Kết quả tham chiếu của tác giả, cùng 20 segment | 0.871¹ | 0.892¹ | 0.915² |

¹ `best_val_in_paper.bin`: bản tốt nhất trong bài báo (detector gốc mạnh hơn `fsd_base`).
² `cyc_result_val_bi_ext_no_empty.bin`: cùng chuỗi `fsd_base` + extend hai chiều + bỏ box rỗng.

Pedestrian và cyclist khớp kết quả của tác giả trong khoảng ±0.3 điểm. Vehicle thấp hơn bản "best" khoảng 1 điểm vì
khác detector gốc. CTRL lấy mẫu ngẫu nhiên tối đa 1024 điểm mỗi frame, nên giữa các lần chạy có dao động khoảng ±0.1 điểm.

## Train trên dữ liệu thật

Cần thêm tập train (khoảng 800 GB tfrecord). Có thể convert tập train theo cùng cách cuốn chiếu (prefix `0`), cùng với
`train_gt.bin`/`fsd6f6e_vehicle_full_train.bin` của tác giả. Train cấu hình gốc (8 GPU × 16 mẫu) trên GPU 8 GB
phải giảm `samples_per_gpu` và chạy lâu hơn nhiều.
