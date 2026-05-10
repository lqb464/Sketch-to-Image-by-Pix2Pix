# Hướng Dẫn Cấu Trúc Repo — Sketch-to-Image (Pix2Pix)

> Tài liệu này giúp các thành viên nhóm hiểu chức năng từng file/folder,  
> biết khi cần gì thì vào đâu, khi thêm file mới thì đặt ở đâu,  
> và quy ước đặt tên để toàn nhóm thống nhất.

---

## Tổng quan cấu trúc

```
Sketch-to-Image-by-Pix2Pix/
│
├── train.py                  ← Điểm vào để train model
├── test.py                   ← Điểm vào để test/inference
├── environment.yml           ← Cài môi trường Conda (local)
├── requirements.txt          ← Cài dependencies pip (Kaggle)
│
├── configs/                  ← File config YAML cho từng experiment
│   ├── baseline_facades.yaml
│   ├── baseline_edges2shoes.yaml
│   └── baseline_cufs.yaml
│
├── data/                     ← Logic load & xử lý dataset
├── datasets/                 ← Script chuẩn bị dữ liệu thô (KHÔNG commit data thô)
├── models/                   ← Kiến trúc model (Generator, Discriminator)
├── options/                  ← Cấu hình tham số train/test
├── scripts/                  ← Script chạy nhanh (train, test, download, eval)
├── util/                     ← Các hàm tiện ích dùng chung
│
├── checkpoints/              ← Model checkpoint sau khi train (KHÔNG commit)
│   └── <tên_experiment>/
│       ├── latest_net_G.pth
│       ├── latest_net_D.pth
│       └── <epoch>_net_G.pth
│
├── results/                  ← Ảnh output sau khi test (KHÔNG commit)
│   └── <tên_experiment>/
│       └── test_latest/
│           └── images/
│               ├── 1_real_A.png    ← Input (sketch)
│               ├── 1_fake_B.png    ← Generated
│               └── 1_real_B.png    ← Ground truth
│
├── figures/                  ← Biểu đồ, figure cho báo cáo
│   ├── loss_curves/
│   ├── qualitative/
│   └── ablation/
│
├── tables/                   ← File CSV kết quả metric
│   └── metrics_<ngày>.csv
│
├── notebooks/                ← Các Kaggle notebooks
│   └── f2-end2end-baseline.ipynb
│
├── report/                   ← File LaTeX báo cáo
│   ├── main.tex
│   └── chapters/
│
└── demo/                     ← App demo Gradio (tạo khi cần)
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
  --name shoes_unet256_lambda100_0524 \
  --model pix2pix \
  --direction AtoB \
  --use_wandb
```

Ví dụ lệnh test:
```bash
python test.py --dataroot ./datasets/edges2shoes \
  --name shoes_unet256_lambda100_0524 \
  --model pix2pix \
  --direction AtoB \
  --num_test 50
```

---

### `configs/` — File config YAML

Mỗi experiment có một file YAML riêng. Thay vì hard-code tham số trong notebook, load config từ đây:

```python
import yaml
with open("./configs/baseline_edges2shoes.yaml") as f:
    cfg = yaml.safe_load(f)
```

Khi làm ablation study, copy file config baseline rồi chỉnh tham số cần thay đổi. Ví dụ:
```
configs/
  baseline_edges2shoes.yaml         ← baseline
  ablation_shoes_unet128.yaml       ← đổi netG: unet_128
  ablation_shoes_lambda50.yaml      ← đổi lambda_L1: 50
```

---

### `data/` — Logic load dataset

| File | Chức năng |
|------|-----------|
| `aligned_dataset.py` | Load dataset dạng ảnh ghép đôi A\|B — **file quan trọng nhất**, dùng cho cả 3 dataset |
| `base_dataset.py` | Class cha chứa logic chung: transform, augmentation |
| `image_folder.py` | Hàm đọc danh sách ảnh từ folder |
| `single_dataset.py` | Load từng ảnh đơn lẻ — dùng khi demo inference |
| `__init__.py` | Tự động chọn dataset class theo `--dataset_mode` |

> **Khi nào đụng vào `data/`?**  
> Khi thêm augmentation, đổi cách load ảnh, hoặc debug lỗi input data.

---

### `datasets/` — Script chuẩn bị dữ liệu thô

| File | Chức năng |
|------|-----------|
| `combine_A_and_B.py` | Ghép 2 folder ảnh riêng biệt thành ảnh đôi A\|B — **bắt buộc dùng cho CUFS** |
| `make_dataset_aligned.py` | Tạo dataset dạng aligned từ dữ liệu thô |
| `download_pix2pix_dataset.sh` | Download dataset `edges2shoes`, `edges2handbags` tự động |

> **Khi nào đụng vào `datasets/`?**  
> Khi chuẩn bị dữ liệu lần đầu. Sau đó thường không cần đụng lại.

---

### `models/` — Kiến trúc model

| File | Chức năng |
|------|-----------|
| `pix2pix_model.py` | **File quan trọng nhất** — forward pass, GAN loss + L1 loss, bước optimize |
| `networks.py` | Kiến trúc chi tiết: UNet Generator, PatchGAN Discriminator, khởi tạo weight |
| `base_model.py` | Class cha: save/load checkpoint, in thông tin model |
| `test_model.py` | Model wrapper đơn giản cho inference (chỉ forward, không tính loss) |
| `__init__.py` | Tự động chọn model class theo `--model` |

> **Khi nào đụng vào `models/`?**  
> Khi làm **ablation study** hoặc **cải tiến kiến trúc** (task M1→M3).

---

### `options/` — Cấu hình tham số

| File | Chức năng |
|------|-----------|
| `base_options.py` | Tham số dùng chung: `--dataroot`, `--model`, `--direction`, `--gpu_ids`... |
| `train_options.py` | Tham số train: `--n_epochs`, `--lr`, `--lambda_L1`, `--batch_size`... |
| `test_options.py` | Tham số test: `--num_test`, `--results_dir`... |

> **Khi nào đụng vào `options/`?**  
> Khi cần thêm tham số mới cho experiment (ví dụ: flag bật/tắt kiến trúc mới).

---

### `scripts/` — Script chạy nhanh

| File | Chức năng |
|------|-----------|
| `train_pix2pix.sh` | Script bash mẫu để train — copy và chỉnh tham số |
| `test_pix2pix.sh` | Script bash mẫu để test |
| `test_single.sh` | Test trên từng ảnh đơn lẻ |
| `download_pix2pix_model.sh` | Download pretrained model có sẵn |
| `eval/metrics.py` | Script tính L1, PSNR, SSIM — output ra CSV vào `tables/` |

> **Khi nào đụng vào `scripts/`?**  
> Khi cần chạy train/test nhanh hoặc tính metric sau khi có kết quả.

---

### `util/` — Tiện ích dùng chung

| File | Chức năng |
|------|-----------|
| `visualizer.py` | Hiển thị và lưu ảnh trong train, tích hợp W&B logging |
| `util.py` | Hàm tiện ích: tensor → ảnh, tạo folder, in thông tin |
| `html.py` | Tạo file HTML xem kết quả dạng gallery |

> **Khi nào đụng vào `util/`?**  
> Khi chỉnh cách log lên W&B (`visualizer.py`) hoặc debug xử lý ảnh.

---

## Quy ước đặt tên

### Experiment
```
<dataset>_<netG>_lambda<lambda_L1>_<ngày MMDD>

Ví dụ:
  facades_unet256_lambda100_0524
  shoes_unet128_lambda50_0525
  cufs_unet256_lambda100_0526
  shoes_unet256_attention_0527    ← experiment cải tiến kiến trúc
```

### Branch
```
main                    ← chỉ merge khi đã kiểm tra kỹ
dev                     ← branch tổng hợp hàng ngày
exp/<mô_tả_ngắn>        ← mỗi experiment một branch riêng

Ví dụ:
  exp/ablation-unet128
  exp/ablation-lambda50
  exp/arch-attention
```

### Figure
```
figures/loss_curves/<tên_experiment>_loss.png
figures/qualitative/<tên_experiment>_samples.png
figures/ablation/<tên_so_sánh>.png
```

### CSV metric
```
tables/metrics_<ngày MMDD>.csv

Cột chuẩn: | experiment | dataset | L1 | PSNR | SSIM | train_time |
```

---

## Thêm file mới vào đâu?

| Loại file mới | Đặt vào đâu |
|---------------|-------------|
| Config YAML cho experiment mới | `configs/` |
| Script preprocess dataset mới | `datasets/` |
| Kiến trúc model mới | `models/` |
| Augmentation mới | `data/base_dataset.py` hoặc `data/aligned_dataset.py` |
| Kaggle notebook | `notebooks/` |
| Figure, chart | `figures/` |
| Kết quả metric CSV | `tables/` |
| File LaTeX | `report/` |
| App demo Gradio | `demo/` |

---

## .gitignore — Những gì KHÔNG commit

```
datasets/
checkpoints/
results/
wandb/
*.pth
*.pkl
__pycache__/
*.pyc
.ipynb_checkpoints/
```

> **Checkpoint** lưu trên **Kaggle Dataset private** — liên hệ Bình để được thêm vào.

---

## Hướng dẫn theo từng thành viên

### Hoàng — Gói C: Train & Ablation Study

**Thường xuyên đụng:**
- `configs/` — copy file config baseline, chỉnh tham số cho từng experiment
- `options/train_options.py` — xem danh sách tham số train có thể dùng
- `checkpoints/` — kiểm tra checkpoint đã lưu chưa
- `results/` — xem ảnh output sau khi test

**Ablation study — chỉ đổi tham số, không cần sửa code:**
- `--netG unet_256` vs `--netG unet_128`
- `--netD basic` vs `--netD pixel` vs `--netD n_layers`
- `--lambda_L1 100` vs `--lambda_L1 50`

**Lưu ý:** Mỗi experiment đặt tên theo quy ước và log lên W&B (`--use_wandb`)

---

### Thi — Gói B + C: Data & Architecture

**Chuẩn bị dữ liệu:**
- `datasets/combine_A_and_B.py` — **bắt buộc dùng cho CUFS** để ghép sketch và ảnh thật
- `datasets/download_pix2pix_dataset.sh` — download `edges2shoes`, `edges2handbags`
- `data/aligned_dataset.py` — xem cách data được load để debug nếu có lỗi

**Cải tiến kiến trúc (M1→M3):**
- `models/networks.py` — file cần sửa khi thêm kiến trúc mới (attention, residual...)
- `models/pix2pix_model.py` — sửa nếu cần thêm loss function mới
- `options/base_options.py` — thêm flag mới để bật/tắt cải tiến kiến trúc

**Lưu ý:** Cải tiến kiến trúc làm trên branch riêng `exp/arch-<tên>`

---

### Trang — Gói D: Báo cáo & Demo

**LaTeX báo cáo:**
- `report/` — toàn bộ file LaTeX
- `figures/` — copy figure, chart từ W&B vào đây để dùng trong LaTeX
- `tables/` — lấy file CSV metric để đưa vào bảng LaTeX

**Script metric:**
- `util/util.py` — có hàm `tensor2im()` hữu ích khi xử lý ảnh

**Demo Gradio:**
- Tạo thư mục `demo/` ở root
- `models/test_model.py` + `data/single_dataset.py` — hai file cần import cho inference
- Có thể dùng pretrained model từ `scripts/download_pix2pix_model.sh` để làm trước, không cần chờ train xong