"""
match_sketch_image.py — Combine sketch+photo thành ảnh ghép A|B và chia train/test

Cấu trúc input (flat, không subfolder):
  sketch/f-001_aug0.jpg  ←→  photo/f-001_aug0.jpg
  sketch/f-001_aug1.jpg  ←→  photo/f-001_aug1.jpg
  ...

Chức năng:
  1. Match từng sketch với photo theo stem (1-1)
  2. Ghép ngang: photo bên trái, sketch bên phải → ảnh 512x256
  3. Shuffle và chia train/test theo photo gốc (tránh data leak:
     f-001_aug0, f-001_aug1, f-001_aug2 luôn cùng tập)
  4. Lưu ra output_dir/train/ và output_dir/test/

Lưu ý: Chạy sau preprocess.py và augmentation.py, TRƯỚC bước train pix2pix.

Usage (tối giản):
  python match_sketch_image.py --sketch_dir ./sketch --photo_dir ./photo

Usage (đầy đủ):
  python match_sketch_image.py \\
    --photo_dir   ./augmented/photo \\
    --sketch_dir  ./augmented/sketch \\
    --output_dir  ./output \\
    --train_ratio 0.8 \\
    --seed        42
"""

import os
import re
import random
import argparse
import cv2

# ───────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Combine sketch+photo và chia train/test — CUFS")
parser.add_argument("--photo_dir",   type=str, default="./photo",
                    help="Thư mục photo (flat, mặc định: ./photo)")
parser.add_argument("--sketch_dir",  type=str, default="./sketch",
                    help="Thư mục sketch (flat, mặc định: ./sketch)")
parser.add_argument("--output_dir",  type=str, default="./output",
                    help="Thư mục output (mặc định: ./output)")
parser.add_argument("--train_ratio", type=float, default=0.8,
                    help="Tỷ lệ train (mặc định: 0.8 → 80/20)")
parser.add_argument("--seed",        type=int,   default=42,
                    help="Random seed (mặc định: 42)")
args = parser.parse_args()

random.seed(args.seed)

PHOTO_ROOT  = args.photo_dir
SKETCH_ROOT = args.sketch_dir
TRAIN_DIR   = os.path.join(args.output_dir, "train")
TEST_DIR    = os.path.join(args.output_dir, "test")
TRAIN_RATIO = args.train_ratio

PHOTO_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
SKETCH_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Build index photo: stem → path ──────────────────────────────────────────
photo_index = {}
for fname in os.listdir(PHOTO_ROOT):
    stem, ext = os.path.splitext(fname)
    if ext.lower() in PHOTO_EXTS:
        photo_index[stem] = os.path.join(PHOTO_ROOT, fname)

# ── Match sketch ↔ photo, gom theo base_id (trước _augN) ────────────────────
# f-001_aug0 → base_id = f-001
def get_base_id(stem: str) -> str:
    return re.sub(r"_aug\d+$", "", stem)

groups = {}   # { base_id: [(sketch_path, photo_path, output_name), ...] }
missing = 0

for fname in sorted(os.listdir(SKETCH_ROOT)):
    stem, ext = os.path.splitext(fname)
    if ext.lower() not in SKETCH_EXTS:
        continue
    if stem not in photo_index:
        print(f"[WARN] Không match được photo cho sketch: {fname}")
        missing += 1
        continue

    sketch_path = os.path.join(SKETCH_ROOT, fname)
    photo_path  = photo_index[stem]
    output_name = stem + ".jpg"
    base_id     = get_base_id(stem)

    if base_id not in groups:
        groups[base_id] = []
    groups[base_id].append((sketch_path, photo_path, output_name))

print(f"[INFO] {sum(len(v) for v in groups.values())} cặp hợp lệ "
      f"từ {len(groups)} base_id, bỏ qua {missing} sketch.")

# ── Shuffle và split theo base_id (tránh data leak) ─────────────────────────
base_ids  = list(groups.keys())
random.shuffle(base_ids)

split_idx = max(1, int(len(base_ids) * TRAIN_RATIO))
train_ids = base_ids[:split_idx]
test_ids  = base_ids[split_idx:]

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

# ── Ghép và lưu ─────────────────────────────────────────────────────────────
total_train = total_test = 0

def combine_and_save(sketch_path, photo_path, output_name, out_dir):
    img    = cv2.imread(photo_path)
    sketch = cv2.imread(sketch_path)
    if img is None or sketch is None:
        return False

    h, w, _ = img.shape
    h_s, w_s, _ = sketch.shape
    new_w_s = int(w_s * h / h_s) if h_s > 0 else w_s
    sketch_resized = cv2.resize(sketch, (new_w_s, h))

    combined = cv2.hconcat([img, sketch_resized])
    cv2.imwrite(os.path.join(out_dir, output_name), combined)
    return True

for base_id in train_ids:
    for sketch_path, photo_path, output_name in groups[base_id]:
        if combine_and_save(sketch_path, photo_path, output_name, TRAIN_DIR):
            total_train += 1

for base_id in test_ids:
    for sketch_path, photo_path, output_name in groups[base_id]:
        if combine_and_save(sketch_path, photo_path, output_name, TEST_DIR):
            total_test += 1

print(f"\n✅ Hoàn tất!")
print(f"   Train : {total_train} ảnh  →  {TRAIN_DIR}")
print(f"   Test  : {total_test}  ảnh  →  {TEST_DIR}")
print(f"   Tỷ lệ : {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}  |  Seed: {args.seed}")
