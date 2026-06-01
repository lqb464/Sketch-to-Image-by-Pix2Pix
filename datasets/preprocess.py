"""
preprocess.py — Preprocessing cho bài toán Sketch-to-Image (Pix2Pix) — CUFS dataset

Cấu trúc CUFS input:
  photos/          ← flat, không có subfolders
    f-001.jpg
    f-002.jpg
    ...
  sketches/
    f-001.jpg      ← cùng tên với photo, quan hệ 1-1
    f-002.jpg
    ...

Chức năng:
  Luồng PHOTO  (độc lập):
    1. Resize về out_size x out_size (mặt đã centered sẵn trong CUFS)
    2. Edge enhancement tùy chọn: dilate / unsharp / both / none

  Luồng SKETCH (độc lập):
    1. Threshold tìm bbox nét vẽ (nền trắng)
    2. Mở rộng bbox thành hình VUÔNG (giữ center, lấy cạnh dài hơn)
    3. Crop vuông → resize về out_size x out_size

  Xử lý lỗi:
    - Sketch không tìm được bbox → bỏ qua cặp, ghi vào failed_sketches.txt

  Lưu ra output_dir/sketch/ và output_dir/photo/ (flat, không subfolder)
  (Bước combine A|B làm sau bằng match_sketch_image.py)

Dependency: pip install opencv-python Pillow

Usage:
  python preprocess.py \\
    --sketch_dir ./sketches \\
    --photo_dir  ./photos \\
    --output_dir ./preprocessed \\
    --enhance    both \\
    --out_size   256
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SKETCH_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
PHOTO_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────
# Matching — flat directory, 1-1 by stem
# ─────────────────────────────────────────────

def find_pairs(sketch_dir: Path, photo_dir: Path):
    """
    Match sketch ↔ photo theo stem (cùng tên file).
    Cả hai folder đều flat (không có subfolders).
    """
    # Build index: stem → path cho photo
    photo_index = {}
    for p in sorted(photo_dir.iterdir()):
        if p.suffix.lower() in PHOTO_EXTS and p.is_file():
            photo_index[p.stem + "-sz1"] = p  # Thêm suffix để tránh trùng tên với sketch

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

    print(f"[INFO] Tìm được {len(pairs)} cặp hợp lệ, bỏ qua {missing} sketch không match.")
    return pairs


# ─────────────────────────────────────────────
# Bbox → Square crop (dùng cho sketch)
# ─────────────────────────────────────────────

def bbox_to_square(x1: int, y1: int, x2: int, y2: int,
                   img_w: int, img_h: int) -> tuple:
    """
    Mở rộng bbox thành hình vuông: lấy cạnh dài hơn, giữ nguyên center.
    Clamp về biên ảnh.
    """
    cx   = (x1 + x2) // 2
    cy   = (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1)
    half = side // 2

    sx1 = max(0, cx - half)
    sy1 = max(0, cy - half)
    sx2 = min(img_w, sx1 + side)
    sy2 = min(img_h, sy1 + side)

    return sx1, sy1, sx2, sy2


# ─────────────────────────────────────────────
# Luồng PHOTO — resize trực tiếp
# ─────────────────────────────────────────────

def process_photo(photo_np: np.ndarray, out_size: int) -> np.ndarray:
    """
    CUFS: mặt đã căn giữa sẵn → resize thẳng về out_size x out_size.
    """
    return cv2.resize(photo_np, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────
# Luồng SKETCH — threshold bbox → square crop
# ─────────────────────────────────────────────

def process_sketch(sketch_np: np.ndarray, out_size: int) -> tuple:
    """
    Tìm bbox nét vẽ bằng threshold → crop vuông → resize.
    Trả về (cropped_np, True) nếu thành công,
            (None, False) nếu không tìm được bbox.
    """
    H, W      = sketch_np.shape[:2]
    gray      = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords    = cv2.findNonZero(binary)

    if coords is None:
        return None, False

    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return None, False

    sx1, sy1, sx2, sy2 = bbox_to_square(x, y, x + w, y + h, W, H)
    crop               = sketch_np[sy1:sy2, sx1:sx2]

    if crop.size == 0:
        return None, False

    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)
    return resized, True


# ─────────────────────────────────────────────
# Edge enhancement — chỉ áp lên PHOTO
# ─────────────────────────────────────────────

def enhance_dilate(photo_np: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Canny edge → dilate → blend nhẹ vào ảnh gốc để làm đậm viền."""
    gray    = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    kernel  = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    edge_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    photo_f  = photo_np.astype(np.float32)
    result   = photo_f * (1.0 - 0.4 * edge_rgb)
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_unsharp(photo_np: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """Unsharp mask trên ảnh màu — làm sắc nét chi tiết."""
    photo_f   = photo_np.astype(np.float32)
    blurred   = cv2.GaussianBlur(photo_f, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(photo_f, 1 + alpha, blurred, -alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_enhancement(photo_np: np.ndarray, mode: str,
                       unsharp_alpha: float = 1.5,
                       dilate_kernel: int = 2) -> np.ndarray:
    """mode: 'dilate' | 'unsharp' | 'both' | 'none'. 'both': unsharp trước, dilate sau."""
    if mode == "none":
        return photo_np
    elif mode == "unsharp":
        return enhance_unsharp(photo_np, unsharp_alpha)
    elif mode == "dilate":
        return enhance_dilate(photo_np, dilate_kernel)
    elif mode == "both":
        return enhance_dilate(enhance_unsharp(photo_np, unsharp_alpha), dilate_kernel)
    else:
        raise ValueError(f"enhance mode không hợp lệ: {mode}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing Sketch-to-Image cho Pix2Pix — CUFS dataset"
    )
    parser.add_argument("--sketch_dir",    type=Path,  required=True,
                        help="Thư mục sketches gốc (flat, không subfolder)")
    parser.add_argument("--photo_dir",     type=Path,  required=True,
                        help="Thư mục photos gốc (flat, không subfolder)")
    parser.add_argument("--output_dir",    type=Path,  required=True,
                        help="Thư mục output (tạo sketch/ và photo/ bên trong)")
    parser.add_argument("--enhance",       type=str,   default="both",
                        choices=["dilate", "unsharp", "both", "none"],
                        help="Edge enhancement áp lên PHOTO (mặc định: both)")
    parser.add_argument("--out_size",      type=int,   default=256,
                        help="Kích thước ảnh output (mặc định: 256)")
    parser.add_argument("--unsharp_alpha", type=float, default=1.5,
                        help="Alpha unsharp mask (mặc định: 1.5)")
    parser.add_argument("--dilate_kernel", type=int,   default=2,
                        help="Kernel size dilate (mặc định: 2)")
    args = parser.parse_args()

    for d, name in [(args.sketch_dir, "sketch_dir"), (args.photo_dir, "photo_dir")]:
        if not d.exists():
            print(f"[ERR] {name} không tồn tại: {d}")
            sys.exit(1)

    print(f"[INFO] sketch_dir  : {args.sketch_dir}")
    print(f"[INFO] photo_dir   : {args.photo_dir}")
    print(f"[INFO] output_dir  : {args.output_dir}")
    print(f"[INFO] enhance     : {args.enhance}  (áp lên photo)")
    print(f"[INFO] out_size    : {args.out_size}x{args.out_size}")
    print(f"[INFO] crop photo  : resize trực tiếp (CUFS đã căn giữa sẵn)")
    print(f"[INFO] crop sketch : threshold → square bbox → resize")
    print()

    pairs = find_pairs(args.sketch_dir, args.photo_dir)
    if not pairs:
        print("[ERR] Không tìm được cặp nào.")
        sys.exit(1)

    out_sketch_dir = args.output_dir / "sketch"
    out_photo_dir  = args.output_dir / "photo"
    out_sketch_dir.mkdir(parents=True, exist_ok=True)
    out_photo_dir.mkdir(parents=True, exist_ok=True)

    failed_sketches = []
    success = fail  = 0

    for i, (sketch_path, photo_path) in enumerate(pairs):
        try:
            sketch_np = np.array(Image.open(sketch_path).convert("RGB"))
            photo_np  = np.array(Image.open(photo_path).convert("RGB"))
        except Exception as e:
            print(f"[ERR] Đọc file: {e}")
            fail += 1
            continue

        # ── Luồng PHOTO ──────────────────────────────────────────────────────
        photo_resized = process_photo(photo_np, args.out_size)
        photo_out     = apply_enhancement(
            photo_resized, args.enhance, args.unsharp_alpha, args.dilate_kernel
        )

        # ── Luồng SKETCH ─────────────────────────────────────────────────────
        sketch_out, sketch_ok = process_sketch(sketch_np, args.out_size)
        if not sketch_ok:
            failed_sketches.append(sketch_path.name)
            fail += 1
            continue

        # ── Lưu output (giữ nguyên tên file gốc) ────────────────────────────
        Image.fromarray(sketch_out).save(out_sketch_dir / sketch_path.name)
        Image.fromarray(photo_out).save(out_photo_dir   / photo_path.name)
        success += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
            print(f"[{i+1}/{len(pairs)}] OK: {success}  Lỗi: {fail}")

    # ── Ghi log lỗi ──────────────────────────────────────────────────────────
    if failed_sketches:
        log = args.output_dir / "failed_sketches.txt"
        log.write_text("\n".join(failed_sketches), encoding="utf-8")
        print(f"\n[WARN] {len(failed_sketches)} sketch không tìm được bbox → {log}")

    print(f"\n✅ Hoàn thành! {success} cặp thành công, {fail} cặp lỗi.")
    print(f"   Output sketch : {out_sketch_dir}")
    print(f"   Output photo  : {out_photo_dir}")


if __name__ == "__main__":
    main()
