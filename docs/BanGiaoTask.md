# Bàn Giao Công Việc — Project Pix2Pix (Sketch-to-Image)

> **Nhóm:** Bình, Hoàng, Thi, Trang  
> **Repo gốc:** https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
> **Repo nhóm:** https://github.com/lqb464/Sketch-to-Image-by-Pix2Pix  
> **Tracking:** Weights & Biases (W&B)

---

## 1. Tổng quan project

Nhóm mình sẽ tái hiện và mở rộng mô hình **pix2pix** cho bài toán **sketch → image** trên 3 dataset:
- `facades`
- `CUFS`
- `edges2shoes`

Kết quả cuối gồm: **baseline đã train**, **ablation study**, **cải tiến kiến trúc nhỏ**, **báo cáo LaTeX** và **demo có UI** (optional).

---

## 2. Phân công tổng thể

| Người | Gói | Trách nhiệm chính |
|-------|-----|-------------------|
| **Bình** | Gói A — Hạ tầng & vận hành | Fork repo, setup baseline notebook Kaggle, W&B tracking, README |
| **Hoàng** | Gói C — Mô hình & thí nghiệm | Train baseline 3 dataset, ablation study |
| **Thi** | Gói B + C (architecture) | Data pipeline, preprocess, EDA, sửa kiến trúc nhỏ |
| **Trang** | Gói D — Báo cáo & Demo | LaTeX skeleton, metric script, error analysis, demo Gradio |

> **Lưu ý:** Trang đang đi làm nên tui ưu tiên cho gói D nha, có thể thực hiện **U1, U2 (demo)** và **L1 (LaTeX skeleton)** trước — hai việc này không phụ thuộc ai và làm được ngay.

---

## 3. Chi tiết việc từng người

### Bình — Gói A: Hạ tầng & Vận hành

| Mã | Việc | Output | Phụ thuộc |
|----|------|--------|-----------|
| F1 | Chuẩn hóa repo | Sơ đồ thư mục, quy ước branch, quy ước đặt tên experiment | — |
| F2 | Kaggle notebook baseline | 1 notebook chạy end-to-end, 1 lệnh baseline chuẩn | F1 |
| F3 | Reproducibility & seed | Seed policy, tên thư mục checkpoint/results, file config mẫu | F1 |
| F4 | Experiment tracker | 1 run log lên W&B, hướng dẫn xem chart | F2 |
| R1 | README chạy trên Kaggle | Setup guide, train/test command, cách nạp dataset | F2, D3 |
| R2 | Thư mục outputs chuẩn | Cấu trúc thư mục thống nhất, naming rules | F3 |

**Làm ngay:** F1 → F4 (đây là việc unlock toàn bộ nhóm)

---

### Hoàng — Gói C: Mô hình & Thí nghiệm

| Mã | Việc | Output | Phụ thuộc |
|----|------|--------|-----------|
| T1 | Baseline trên edges2shoes | Checkpoint tốt nhất, metric test, bộ ảnh kết quả | F2, F3, F4, D3 |
| T2 | Baseline trên edges2handbags | Checkpoint, metric, sample results | T1 hoặc D3 |
| A1 | So sánh unet_256 vs unet_128 | Bảng metric, training time, nhận xét | T1 |
| A2 | So sánh basic vs n_layers vs pixel discriminator | Bảng metric, ảnh định tính, nhận xét artifact | T1 |
| A3 | So sánh lambda_L1 | Bảng metric, ảnh (ít nhất 2 giá trị lambda) | T1 |
| A4 | Tuning tài nguyên Kaggle | Batch size, load/crop size hợp lý, thời gian train/epoch | T1 |

**Làm ngay:** Chờ F2 xong → bắt đầu T1 ngay

---

### Thi — Gói B + C (Architecture)

| Mã | Việc | Output | Phụ thuộc |
|----|------|--------|-----------|
| D1 | Chốt 3 dataset chính thức | Bảng dataset, lý do chọn, link tải | — |
| D2 | Tải và kiểm tra từng dataset | Folder gốc, số lượng ảnh, kích thước, vấn đề dữ liệu | D1 |
| D3 | Pipeline preprocess → format pix2pix | Script/notebook preprocess, folder output chuẩn hóa | D2 |
| D4 | Dataset EDA & visualization | Bảng train/val/test, 6–12 ảnh A/B, nhận xét độ khó | D3 |
| D5 | Data augmentation study | Danh sách augmentation an toàn, 1 thí nghiệm ngắn nếu cần | D3 |
| T3 | Baseline trên CUFS/CUFSF | Checkpoint, metric, sample results | D3 |
| M1 | Chọn hướng sửa kiến trúc | Decision note, phạm vi file cần sửa | T1 |
| M2 | Cài đặt cải tiến kiến trúc | Code chạy được, option bật/tắt | M1 |
| M3 | Train & so sánh với baseline | Metric so sánh, ảnh so sánh, kết luận giữ hay bỏ | M2 |

**Làm ngay:** D1 và D2 (không phụ thuộc ai, làm được ngay hôm nay)

---

### Trang — Gói D: Báo cáo & Demo

| Mã | Việc | Output | Phụ thuộc |
|----|------|--------|-----------|
| L1 | Dựng skeleton LaTeX 5 chương | file main.tex, chapter files, bib file | — |
| U1 | Chốt scope demo Gradio | Quyết định tính năng tối thiểu | T1 |
| U2 | Demo inference | App demo chạy local hoặc notebook | U1, checkpoint baseline |
| U3 | Chuẩn bị assets demo | Gallery input/output, script demo | U2 |
| E1 | Script metric | Script tính L1, PSNR, SSIM (LPIPS nếu kịp), file CSV | T1 |
| E2 | Tổng hợp bảng kết quả | Bảng metric, bảng thời gian train | E1, A1, A2, A3, M3 |
| E3 | Error analysis | Figure good/bad cases, phân loại lỗi | T1, T2, T3 |
| E4 | Visualization training curves | Chart loss, chart metric, nhận xét hội tụ | F4, T1 |
| L2 | Viết Chương 1 & 2 | Draft mô tả bài toán, pix2pix, loss, kiến trúc | L1 |
| L3 | Viết Chương 3 | Draft dữ liệu, preprocess, pipeline + bảng dataset | D4, L1 |
| L4 | Viết Chương 4 | Draft thực nghiệm, bảng và hình hoàn chỉnh | E2, E3, E4, L1 |
| L5 | Viết Chương 5 & bảng phân công | Draft kết luận, hạn chế, đóng góp từng người | Gần hoàn thành |

**Làm ngay:** L1 và U1 — cả hai không cần chờ ai

---

## 4. Quy ước repo

### Branch
```
main          ← chỉ merge khi đã kiểm tra
dev           ← branch tổng hợp hàng ngày
exp/<tên>     ← mỗi experiment một branch riêng
               ví dụ: exp/ablation-unet128, exp/arch-attention
```

### Thư mục nhóm sẽ thêm vào repo
```
datasets/         ← dữ liệu (không commit lên git, dùng .gitignore)
checkpoints/      ← model checkpoint
results/          ← ảnh output test
figures/          ← biểu đồ, figure cho báo cáo
tables/           ← CSV kết quả metric
report/           ← file LaTeX
notebooks/        ← Kaggle notebooks
scripts/          ← script preprocess, metric
```

### Đặt tên experiment
```
<dataset>_<model>_<thay_đổi>_<ngày>
ví dụ: shoes_unet256_lambda100_0524
        cufs_unet128_lambda50_0525
```

---

## 5. Dependency & Critical Path

```
Bình:   F1 ──→ F2 ──→ F3 ──→ F4
                │
                ▼
Hoàng:         T1 ──→ T2
                │
                └──→ A1 / A2 / A3 / A4

Thi:   D1 ──→ D2 ──→ D3 ──→ D4
                        │
                        └──→ D5
                        └──→ T3
                        └──→ M1 ──→ M2 ──→ M3

Trang: L1 (ngay) ──────────────────────→ L2 → L3 → L4 → L5
       U1 (ngay) ──→ U2 ──→ U3
               (chờ checkpoint T1) ──→ E1 ──→ E2 / E3 / E4
```

---

## 6. Các mốc bàn giao nội bộ

| Mốc | Điều kiện hoàn thành |
|-----|----------------------|
| **Mốc 1** — Repo chạy được | F1, F2, F3 xong + 1 run baseline ngắn |
| **Mốc 2** — Dữ liệu hoàn chỉnh | D1–D4 xong, 3 dataset đúng format |
| **Mốc 3** — Baseline xong | T1 xong, T2 hoặc T3 có ít nhất 1 run, E1 dùng được |
| **Mốc 4** — Kết quả so sánh | A1–A3 xong, M3 xong, E2/E3/E4 gần hoàn chỉnh |
| **Mốc 5** — Chốt nộp | L1–L5 hoàn tất, README hoàn tất, demo optional |

---

## 7. Lưu ý chung

- **Không commit dataset** lên repo — dùng `.gitignore` cho thư mục `datasets/`
- **Mọi experiment** phải log lên W&B (xem hướng dẫn ở F4)
- **Checkpoint** đặt tên theo quy ước trên, lưu vào `checkpoints/<tên_experiment>/`
- **Khi có kết quả mới** → báo vào nhóm Messenger + push lên branch tương ứng
- **F2 (notebook baseline)** là việc quan trọng nhất giai đoạn đầu — Hoàng và Thi chờ F2 xong để bắt đầu được
