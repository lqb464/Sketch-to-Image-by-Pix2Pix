"""
augmentation.py — Augmentation cho bài toán Sketch-to-Image (Pix2Pix) — CUFS dataset

Áp lên từng cặp sketch+photo đã qua preprocess, TRƯỚC bước combine.
Ảnh đầu vào đã là 256x256 — không crop thêm.

Cấu trúc input (flat, không subfolder):
  preprocessed/sketch/f-001.jpg
  preprocessed/photo/f-001.jpg

Transforms:
  Đồng bộ cả hai : horizontal flip (prob)
  Đồng bộ cả hai : random rotation (góc random liên tục trong ±max_deg)
  Chỉ photo      : gaussian blur   (sigma random trong [0.1, max_blur_sigma])
  Chỉ photo      : gaussian noise  (std random trong [1, max_noise_std])
  Chỉ photo      : color jitter    (brightness / contrast / saturation)

Mỗi cặp input sinh ra N cặp augmented (--aug_per_image).
aug0 luôn là bản gốc (không transform).

Usage:
  python augmentation.py \\
    --sketch_dir    ./preprocessed/sketch \\
    --photo_dir     ./preprocessed/photo \\
    --output_dir    ./augmented \\
    --aug_per_image 3 \\
    --flip_prob     0.5 \\
    --rotation_deg  15 \\
    --blur_sigma    1.5 \\
    --noise_std     10 \\
    --jitter_strength 0.1 \\
    --seed          42
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SKETCH_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
PHOTO_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────
# Matching — flat directory, 1-1 by stem
# ─────────────────────────────────────────────

def find_pairs(sketch_dir: Path, photo_dir: Path):
    """Match sketch ↔ photo theo stem trong flat directory."""
    photo_index = {}
    for p in sorted(photo_dir.iterdir()):
        if p.suffix.lower() in PHOTO_EXTS and p.is_file():
            photo_index[p.stem + "-sz1"] = p

    pairs   = []
    missing = 0

    for sketch_path in sorted(sketch_dir.iterdir()):
        if sketch_path.suffix.lower() not in SKETCH_EXTS or not sketch_path.is_file():
            continue
        if sketch_path.stem in photo_index:
            pairs.append((sketch_path, photo_index[sketch_path.stem]))
        else:
            print(f"[WARN] Không match được photo cho sketch: {sketch_path.name}")
            missing += 1

    print(f"[INFO] Tìm được {len(pairs)} cặp, bỏ qua {missing} sketch không match.")
    return pairs


# ─────────────────────────────────────────────
# Transforms đồng bộ A+B
# ─────────────────────────────────────────────

def sync_flip(sketch_np: np.ndarray,
              photo_np: np.ndarray,
              prob: float) -> tuple:
    """Horizontal flip đồng bộ cả hai."""
    if random.random() < prob:
        sketch_np = np.fliplr(sketch_np).copy()
        photo_np  = np.fliplr(photo_np).copy()
    return sketch_np, photo_np


def sync_rotation(sketch_np: np.ndarray,
                  photo_np: np.ndarray,
                  max_deg: float) -> tuple:
    """
    Rotate đồng bộ cả hai với góc random liên tục trong [-max_deg, +max_deg].
    Sketch: fill nền trắng. Photo: fill BORDER_REFLECT để tránh viền đen.
    """
    angle = random.uniform(-max_deg, max_deg)
    H, W  = sketch_np.shape[:2]
    M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)

    sketch_rot = cv2.warpAffine(
        sketch_np, M, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    photo_rot = cv2.warpAffine(
        photo_np, M, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return sketch_rot, photo_rot


# ─────────────────────────────────────────────
# Transforms chỉ áp lên PHOTO
# ─────────────────────────────────────────────

def apply_blur(photo_np: np.ndarray, max_sigma: float) -> np.ndarray:
    """Gaussian blur với sigma random trong [0.1, max_sigma]."""
    sigma = random.uniform(0.1, max_sigma)
    return cv2.GaussianBlur(photo_np, (0, 0), sigmaX=sigma)


def apply_noise(photo_np: np.ndarray, max_std: float) -> np.ndarray:
    """Gaussian noise với std random trong [1, max_std]."""
    std   = random.uniform(1.0, max_std)
    noise = np.random.normal(0, std, photo_np.shape).astype(np.float32)
    return np.clip(photo_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_color_jitter(photo_np: np.ndarray, strength: float) -> np.ndarray:
    """Color jitter: brightness / contrast / saturation. strength 0.1=nhẹ, 0.3=mạnh."""
    img = Image.fromarray(photo_np)
    for Enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        factor = 1.0 + random.uniform(-strength, strength)
        img    = Enhancer(img).enhance(factor)
    return np.array(img)


# ─────────────────────────────────────────────
# Core: tạo 1 augmented version
# ─────────────────────────────────────────────

def augment_pair(sketch_np: np.ndarray,
                 photo_np: np.ndarray,
                 aug_index: int,
                 flip_prob: float,
                 rotation_deg: float,
                 blur_sigma: float,
                 noise_std: float,
                 jitter_strength: float) -> tuple:
    """
    aug_index == 0 : bản gốc, không transform.
    aug_index >  0 : áp toàn bộ transforms.
    """
    if aug_index == 0:
        return sketch_np, photo_np

    # 1. Flip đồng bộ
    sketch_np, photo_np = sync_flip(sketch_np, photo_np, flip_prob)

    # 2. Rotation đồng bộ
    if rotation_deg > 0:
        sketch_np, photo_np = sync_rotation(sketch_np, photo_np, rotation_deg)

    # 3. Blur — chỉ photo
    if blur_sigma > 0:
        photo_np = apply_blur(photo_np, blur_sigma)

    # 4. Noise — chỉ photo
    if noise_std > 0:
        photo_np = apply_noise(photo_np, noise_std)

    # 5. Color jitter — chỉ photo
    if jitter_strength > 0:
        photo_np = apply_color_jitter(photo_np, jitter_strength)

    return sketch_np, photo_np


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Augmentation Sketch-to-Image cho Pix2Pix — CUFS dataset"
    )
    parser.add_argument("--sketch_dir",      type=Path,  required=True,
                        help="Thư mục sketch đã preprocess (flat)")
    parser.add_argument("--photo_dir",       type=Path,  required=True,
                        help="Thư mục photo đã preprocess (flat)")
    parser.add_argument("--output_dir",      type=Path,  required=True,
                        help="Thư mục output augmented")
    parser.add_argument("--aug_per_image",   type=int,   default=3,
                        help="Số cặp sinh ra mỗi ảnh gốc, bao gồm aug0=bản gốc (mặc định: 3)")
    parser.add_argument("--flip_prob",       type=float, default=0.5,
                        help="Xác suất horizontal flip (mặc định: 0.5)")
    parser.add_argument("--rotation_deg",    type=float, default=15.0,
                        help="Góc rotation tối đa ±deg (mặc định: 15, đặt 0 để tắt)")
    parser.add_argument("--blur_sigma",      type=float, default=1.5,
                        help="Gaussian blur sigma tối đa, chỉ photo (mặc định: 1.5, đặt 0 để tắt)")
    parser.add_argument("--noise_std",       type=float, default=10.0,
                        help="Gaussian noise std tối đa, chỉ photo (mặc định: 10, đặt 0 để tắt)")
    parser.add_argument("--jitter_strength", type=float, default=0.1,
                        help="Cường độ color jitter, chỉ photo (mặc định: 0.1, đặt 0 để tắt)")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Random seed (mặc định: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for d, name in [(args.sketch_dir, "sketch_dir"), (args.photo_dir, "photo_dir")]:
        if not d.exists():
            print(f"[ERR] {name} không tồn tại: {d}")
            sys.exit(1)

    print(f"[INFO] sketch_dir      : {args.sketch_dir}")
    print(f"[INFO] photo_dir       : {args.photo_dir}")
    print(f"[INFO] output_dir      : {args.output_dir}")
    print(f"[INFO] aug_per_image   : {args.aug_per_image}  (aug0 = bản gốc)")
    print(f"[INFO] flip_prob       : {args.flip_prob}")
    print(f"[INFO] rotation_deg    : ±{args.rotation_deg}°  (random liên tục)")
    print(f"[INFO] blur_sigma      : [0.1, {args.blur_sigma}]  (chỉ photo)")
    print(f"[INFO] noise_std       : [1, {args.noise_std}]     (chỉ photo)")
    print(f"[INFO] jitter_strength : {args.jitter_strength}    (chỉ photo)")
    print(f"[INFO] seed            : {args.seed}")
    print()

    pairs = find_pairs(args.sketch_dir, args.photo_dir)
    if not pairs:
        print("[ERR] Không tìm được cặp nào.")
        sys.exit(1)

    out_sketch_dir = args.output_dir / "sketch"
    out_photo_dir  = args.output_dir / "photo"
    out_sketch_dir.mkdir(parents=True, exist_ok=True)
    out_photo_dir.mkdir(parents=True, exist_ok=True)

    total           = len(pairs) * args.aug_per_image
    success = fail  = 0

    for pair_idx, (sketch_path, photo_path) in enumerate(pairs):
        try:
            sketch_np = np.array(Image.open(sketch_path).convert("RGB"))
            photo_np  = np.array(Image.open(photo_path).convert("RGB"))
        except Exception as e:
            print(f"[ERR] Đọc file ({sketch_path.name}): {e}")
            fail += args.aug_per_image
            continue

        stem      = sketch_path.stem         # e.g. f-001
        sk_ext    = sketch_path.suffix       # e.g. .jpg
        ph_ext    = photo_path.suffix

        for aug_i in range(args.aug_per_image):
            try:
                aug_sketch, aug_photo = augment_pair(
                    sketch_np.copy(), photo_np.copy(),
                    aug_index       = aug_i,
                    flip_prob       = args.flip_prob,
                    rotation_deg    = args.rotation_deg,
                    blur_sigma      = args.blur_sigma,
                    noise_std       = args.noise_std,
                    jitter_strength = args.jitter_strength,
                )

                # Tên: f-001_aug0.jpg, f-001_aug1.jpg, ...
                Image.fromarray(aug_sketch).save(out_sketch_dir / f"{stem}_aug{aug_i}{sk_ext}")
                Image.fromarray(aug_photo).save(out_photo_dir   / f"{stem}_aug{aug_i}{ph_ext}")
                success += 1

            except Exception as e:
                print(f"[ERR] aug{aug_i} của {sketch_path.name}: {e}")
                fail += 1

        done = (pair_idx + 1) * args.aug_per_image
        if done % 300 == 0 or (pair_idx + 1) == len(pairs):
            print(f"[{done}/{total}] OK: {success}  Lỗi: {fail}")

    print(f"\n✅ Hoàn thành! {success} ảnh thành công, {fail} ảnh lỗi.")
    print(f"   Tổng cặp gốc  : {len(pairs)}")
    print(f"   Tổng ảnh sinh : {success}")
    print(f"   Output sketch : {out_sketch_dir}")
    print(f"   Output photo  : {out_photo_dir}")


if __name__ == "__main__":
    main()
