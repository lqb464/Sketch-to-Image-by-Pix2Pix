# Hướng Dẫn Cấu Trúc Repo — Sketch-to-Image (Pix2Pix)

> Tài liệu này là cho các thành viên nhóm hiểu chức năng từng file/folder,  
> biết khi cần gì thì vào đâu, và khi thêm file mới thì đặt ở đâu.

---

## Tổng quan cấu trúc

```
repo/
├── train.py                  ← Điểm vào để train model
├── test.py                   ← Điểm vào để test/inference
├── environment.yml           ← Cài môi trường Conda
├── requirements.txt          ← Cài dependencies pip
│
├── data/                     ← Logic load & xử lý dataset
├── datasets/                 ← Script chuẩn bị dữ liệu thô
├── models/                   ← Kiến trúc model (Generator, Discriminator)
├── options/                  ← Cấu hình tham số train/test
├── scripts/                  ← Script chạy nhanh (train, test, download)
├── util/                     ← Các hàm tiện ích dùng chung
│
├── checkpoints/              ← Model checkpoint sau khi train (không commit lên git)
├── results/                  ← Ảnh output sau khi test
├── figures/                  ← Biểu đồ, figure cho báo cáo
├── tables/                   ← File CSV kết quả metric
├── notebooks/                ← Các file notebooks thử nghiệm
└── report/                   ← File LaTeX báo cáo
```

---

## Chi tiết từng file/folder

### `train.py` và `test.py`
Hai file điểm vào chính của toàn bộ project.

- `train.py` — Chạy để **train model**. Nhận tham số từ `options/`, load data từ `data/`, khởi tạo model từ `models/`, log kết quả qua W&B.
- `test.py` — Chạy để **test/inference**. Load checkpoint từ `checkpoints/`, chạy model trên tập test, lưu ảnh output vào `results/`.

Ví dụ lệnh train:
```bash
python train.py --dataroot ./datasets/edges2shoes \
  --name shoes_unet256_lambda100_0610 \
  --model pix2pix \
  --direction AtoB
```

Ví dụ lệnh test:
```bash
python test.py --dataroot ./datasets/edges2shoes \
  --name shoes_unet256_lambda100_0610 \
  --model pix2pix \
  --direction AtoB \
  --num_test 50
```

---

### `data/` — Logic load dataset

| File | Chức năng |
|------|-----------|
| `aligned_dataset.py` | Load dataset dạng ảnh ghép đôi A\|B — **file quan trọng nhất**, dùng cho cả 3 dataset của nhóm |
| `base_dataset.py` | Class cha chứa logic chung cho mọi dataset (transform, augmentation) |
| `image_folder.py` | Hàm tiện ích để đọc danh sách ảnh từ folder |
| `single_dataset.py` | Load từng ảnh đơn lẻ (không cần cặp A/B) — dùng khi demo inference |
| `__init__.py` | Tự động chọn dataset class phù hợp dựa vào tham số `--dataset_mode` |

> **Khi nào đụng vào `data/`?**  
> Khi cần thêm augmentation, đổi cách load ảnh, hoặc debug lỗi liên quan đến input data.

---

### `datasets/` — Script chuẩn bị dữ liệu thô

| File | Chức năng |
|------|-----------|
| `combine_A_and_B.py` | Ghép 2 folder ảnh riêng biệt (A và B) thành ảnh đôi A\|B — **cần thiết cho CUFS** |
| `make_dataset_aligned.py` | Tạo dataset dạng aligned từ dữ liệu thô |
| `download_pix2pix_dataset.sh` | Download dataset `edges2shoes`, `edges2handbags` về tự động |

> **Khi nào đụng vào `datasets/`?**  
> Khi chuẩn bị dữ liệu lần đầu — download, ghép ảnh, preprocess. Sau đó thường không cần đụng lại.

---

### `models/` — Kiến trúc model

| File | Chức năng |
|------|-----------|
| `pix2pix_model.py` | **File quan trọng nhất** — định nghĩa toàn bộ model pix2pix: forward pass, loss function (GAN loss + L1 loss), bước optimize |
| `networks.py` | Định nghĩa kiến trúc chi tiết: Generator (UNet), Discriminator (PatchGAN), hàm khởi tạo weight |
| `base_model.py` | Class cha chứa logic chung: save/load checkpoint, in thông tin model |
| `test_model.py` | Model wrapper đơn giản dùng khi inference (chỉ forward, không tính loss) |
| `__init__.py` | Tự động chọn model class dựa vào tham số `--model` |

> **Khi nào đụng vào `models/`?**  
> Khi làm **ablation study** (đổi generator/discriminator) hoặc **cải tiến kiến trúc** (task M1→M3).

---

### `options/` — Cấu hình tham số

| File | Chức năng |
|------|-----------|
| `base_options.py` | Tham số dùng chung cho cả train lẫn test: `--dataroot`, `--model`, `--direction`, `--gpu_ids`... |
| `train_options.py` | Tham số riêng cho train: `--n_epochs`, `--lr`, `--lambda_L1`, `--batch_size`... |
| `test_options.py` | Tham số riêng cho test: `--num_test`, `--results_dir`... |

> **Khi nào đụng vào `options/`?**  
> Khi cần thêm tham số mới cho experiment (ví dụ thêm flag bật/tắt kiến trúc mới).

---

### `scripts/` — Script chạy nhanh

| File | Chức năng |
|------|-----------|
| `train_pix2pix.sh` | Script bash mẫu để train — copy và chỉnh tham số cho từng experiment |
| `test_pix2pix.sh` | Script bash mẫu để test |
| `test_single.sh` | Script test trên từng ảnh đơn lẻ |
| `download_pix2pix_model.sh` | Download pretrained model có sẵn |
| `conda_deps.sh` / `install_deps.sh` | Cài môi trường trên Linux — **nhóm dùng Kaggle nên không cần** |

> **Khi nào đụng vào `scripts/`?**  
> Khi cần chạy train/test nhanh. Copy file `.sh` làm template, đổi tên experiment và tham số.

---

### `util/` — Tiện ích dùng chung

| File | Chức năng |
|------|-----------|
| `visualizer.py` | Hiển thị và lưu ảnh trong quá trình train, tích hợp W&B logging |
| `util.py` | Các hàm tiện ích: xử lý tensor → ảnh, tạo folder, in thông tin |
| `html.py` | Tạo file HTML để xem kết quả ảnh output dạng gallery |
| `get_data.py` | Script download dataset (cũ, thay thế bằng `datasets/download_pix2pix_dataset.sh`) |

> **Khi nào đụng vào `util/`?**  
> Khi cần chỉnh cách log lên W&B (`visualizer.py`) hoặc debug cách xử lý ảnh.

---

### Các thư mục output (không commit lên git trừ `.gitkeep`)

| Thư mục | Lưu gì | Đặt tên theo quy ước |
|---------|--------|----------------------|
| `checkpoints/` | Model checkpoint sau mỗi epoch | `<dataset>_<model>_<thay_đổi>_<ngày>/` |
| `results/` | Ảnh output sau khi chạy `test.py` | Tự động tạo theo tên experiment |
| `figures/` | Biểu đồ loss, metric, figure cho báo cáo | Đặt tên mô tả rõ ràng |
| `tables/` | File CSV kết quả metric (L1, PSNR, SSIM) | Đặt tên theo ablation/experiment |
| `notebooks/` | Kaggle notebooks | Đặt tên theo task (ví dụ: `train_baseline_shoes.ipynb`) |
| `report/` | File LaTeX báo cáo | Theo cấu trúc LaTeX skeleton |

---

## Quy ước đặt tên experiment

```
<dataset>_<model>_<thay_đổi>_<ngày>

Ví dụ:
  shoes_unet256_lambda100_0610
  handbags_unet128_lambda50_0612
  cufs_unet256_baseline_0615
  shoes_unet256_attention_0618    ← experiment cải tiến kiến trúc
```

---

## Thêm file mới vào đâu?

| Loại file mới | Đặt vào đâu |
|---------------|-------------|
| Script preprocess dataset mới | `datasets/` |
| Script tính metric (L1, PSNR, SSIM) | `scripts/` hoặc tạo thêm thư mục `scripts/eval/` |
| Kiến trúc model mới / sửa model | `models/` |
| Augmentation mới | `data/base_dataset.py` hoặc `data/aligned_dataset.py` |
| Kaggle notebook | `notebooks/` |
| Figure, chart | `figures/` |
| Kết quả metric CSV | `tables/` |
| File LaTeX | `report/` |
| App demo Gradio | Tạo thư mục mới `demo/` ở root |

---

## Hướng dẫn theo từng thành viên

### Hoàng — Gói C: Train & Ablation Study

**Thường xuyên đụng:**
- `scripts/train_pix2pix.sh` — copy làm template cho mỗi experiment, đổi tham số
- `options/train_options.py` — xem danh sách tham số train có thể dùng
- `checkpoints/` — kiểm tra checkpoint đã lưu chưa
- `results/` — xem ảnh output sau khi test

**Ablation study:**
- `options/train_options.py` — tham số `--netG` (unet_256 vs unet_128), `--netD` (basic vs pixel), `--lambda_L1`
- Không cần sửa code — chỉ đổi tham số khi chạy lệnh

**Lưu ý:** Mỗi experiment phải đặt tên theo quy ước và log lên W&B (`--use_wandb`)

---

### Thi — Gói B + C: Data & Architecture

**Chuẩn bị dữ liệu:**
- `datasets/combine_A_and_B.py` — **bắt buộc dùng cho CUFS** để ghép ảnh sketch và ảnh thật
- `datasets/download_pix2pix_dataset.sh` — download `edges2shoes`, `edges2handbags`
- `data/aligned_dataset.py` — xem cách data được load để debug nếu có lỗi

**Cải tiến kiến trúc (M1→M3):**
- `models/networks.py` — định nghĩa UNet Generator và PatchGAN Discriminator, đây là file cần sửa khi thêm kiến trúc mới (ví dụ: attention, residual block)
- `models/pix2pix_model.py` — sửa nếu cần thêm loss function mới
- `options/base_options.py` — thêm flag mới để bật/tắt cải tiến kiến trúc

**Lưu ý:** Cải tiến kiến trúc nên làm trên branch riêng `exp/arch-<tên>`

---

### Trang — Gói D: Báo cáo & Demo

**LaTeX báo cáo:**
- `report/` — toàn bộ file LaTeX đặt ở đây
- `figures/` — copy figure, chart từ W&B hoặc tự tạo vào đây để dùng trong LaTeX
- `tables/` — lấy file CSV metric từ đây để đưa vào bảng LaTeX

**Script metric:**
- Tạo file mới `scripts/eval/metrics.py` — tính L1, PSNR, SSIM trên folder `results/`
- `util/util.py` — có hàm `tensor2im()` hữu ích khi xử lý ảnh để tính metric

**Demo Gradio:**
- Tạo thư mục mới `demo/` ở root
- `models/test_model.py` + `data/single_dataset.py` — hai file cần import khi viết inference cho demo
- `checkpoints/` — load checkpoint tốt nhất từ đây

**Lưu ý:** Demo không cần chờ toàn bộ train xong — có thể dùng pretrained model download từ `scripts/download_pix2pix_model.sh` để làm trước
