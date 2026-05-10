# Sketch-to-Image bằng Pix2Pix

Repo của nhóm tái hiện và mở rộng mô hình **pix2pix** cho bài toán **sketch → image** trên 3 dataset: `facades`, `edges2shoes`, `CUFS`.

- **Repo gốc:** https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- **W&B tracking:** https://wandb.ai/ *(liên hệ Bình để được thêm vào project)*
- **Checkpoint:** Kaggle Dataset private *(liên hệ Bình để được thêm vào)*

---

## 1. Setup trên Kaggle

### Bước 1 — Tạo notebook mới trên Kaggle
- Vào [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
- Bật GPU: **Settings → Accelerator → GPU T4 x2**
- Bật Internet: **Settings → Internet → On**

### Bước 2 — Thêm W&B API Key vào Kaggle Secrets
- Vào **Add-ons → Secrets → Add a new secret**
- Name: `WANDB_API_KEY` | Value: API key của bạn (lấy tại https://wandb.ai/settings)

### Bước 3 — Thêm dataset vào notebook
- **Add Data** → tìm `sketch2image-dataset` (của lqb464) → Add
- **Add Data** → tìm `pix2pix-checkpoints` (của lqb464) → Add *(nếu cần dùng checkpoint)*

### Bước 4 — Clone repo và cài dependencies
```python
!git clone -b dev https://github.com/lqb464/Sketch-to-Image-by-Pix2Pix

import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")

os.chdir('Sketch-to-Image-by-Pix2Pix/')
!pip install -q -r requirements.txt
```

---

## 2. Cấu trúc dataset

Dataset được mount tại `/kaggle/input/datasets/lqb464/sketch2image-dataset/`:

```
sketch2image-dataset/
├── facades/
│   ├── train/      ← ảnh AB ghép (sketch + photo)
│   ├── val/
│   └── test/
├── edges2shoes/
│   ├── train/
│   └── val/
└── cufs/
    ├── train/
    └── test/
```

> Mỗi ảnh là ảnh ghép **A|B** (sketch bên trái, photo bên phải), kích thước 512×256. 
> Riêng bộ facades thì sketch bên phải, photo bên trái

---

## 3. Train

### Dùng config YAML (khuyến nghị)
```python
import yaml, random, numpy as np, torch

CONFIG_PATH = "./configs/baseline_facades.yaml"
# CONFIG_PATH = "./configs/baseline_edges2shoes.yaml"
# CONFIG_PATH = "./configs/baseline_cufs.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Fix seed
SEED = cfg.get("seed", 42)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
```

Sau đó chạy train:
```bash
!python train.py \
  --dataroot {cfg['dataroot']} \
  --name {cfg['name']} \
  --model pix2pix \
  --direction {cfg['direction']} \
  --batch_size {cfg['batch_size']} \
  --n_epochs {cfg['n_epochs']} \
  --n_epochs_decay {cfg['n_epochs_decay']} \
  --use_wandb
```

### Lệnh train thủ công (từng dataset)
```bash
# Facades
!python train.py \
  --dataroot /kaggle/input/datasets/lqb464/sketch2image-dataset/facades \
  --name facades_unet256_lambda100_0524 \
  --model pix2pix --direction BtoA \
  --batch_size 16 --lr 0.0004 \
  --n_epochs 100 --n_epochs_decay 100 \
  --use_wandb --no_html

# edges2shoes
!python train.py \
  --dataroot /kaggle/input/datasets/lqb464/sketch2image-dataset/edges2shoes \
  --name shoes_unet256_lambda100_0524 \
  --model pix2pix --direction AtoB \
  --batch_size 16 --lr 0.0004 \
  --n_epochs 100 --n_epochs_decay 100 \
  --use_wandb --no_html

# CUFS
!python train.py \
  --dataroot /kaggle/input/datasets/lqb464/sketch2image-dataset/cufs \
  --name cufs_unet256_lambda100_0524 \
  --model pix2pix --direction AtoB \
  --batch_size 8 --lr 0.0004 \
  --n_epochs 100 --n_epochs_decay 100 \
  --use_wandb --no_html
```

---

## 4. Test

```bash
# Facades
!python test.py \
  --dataroot /kaggle/input/datasets/lqb464/sketch2image-dataset/facades \
  --name facades_unet256_lambda100_0524 \
  --model pix2pix --direction BtoA \
  --num_test 20 --use_wandb

# edges2shoes
!python test.py \
  --dataroot /kaggle/input/datasets/lqb464/sketch2image-dataset/edges2shoes \
  --name shoes_unet256_lambda100_0524 \
  --model pix2pix --direction AtoB \
  --num_test 20 --use_wandb

# CUFS
!python test.py \
  --dataroot /kaggle/input/datasets/lqb464/sketch2image-dataset/cufs \
  --name cufs_unet256_lambda100_0524 \
  --model pix2pix --direction AtoB \
  --num_test 20 --use_wandb
```

---

## 5. Quy ước đặt tên experiment

```
<dataset>_<netG>_lambda<lambda_L1>_<ngày>

Ví dụ:
  facades_unet256_lambda100_0524
  shoes_unet128_lambda50_0525
  cufs_unet256_lambda100_0526
```

---

## 6. Quy ước branch

```
main          ← chỉ merge khi đã kiểm tra kỹ
dev           ← branch tổng hợp hàng ngày
exp/<tên>     ← mỗi experiment một branch riêng

Ví dụ:
  exp/ablation-unet128
  exp/ablation-lambda50
  exp/arch-attention
```

---

## 7. Lưu ý

- **Không commit dataset** lên repo — `datasets/` đã có trong `.gitignore`
- **Checkpoint** lưu trên Kaggle Dataset private, không push lên GitHub
- **Mọi experiment** phải log lên W&B
- Khi có kết quả mới → báo vào nhóm Messenger + push lên branch tương ứng
