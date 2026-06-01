"""
run_pipeline.py — Full pipeline: Preprocess → Augmentation → Combine + Split

Dành cho CUFS dataset (flat directory, 1-1 matching, không có class subfolders).

Chạy tuần tự 3 bước:
  1. preprocess.py         : resize photo + edge enhancement, threshold square crop sketch
  2. augmentation.py       : flip / rotation đồng bộ + blur / noise / jitter chỉ photo
  3. match_sketch_image.py : combine A|B ghép ngang, chia train/test

Dependency: pip install opencv-python Pillow

Usage (tối giản):
  python run_pipeline.py --sketch_dir ./sketches --photo_dir ./photos

Usage (đầy đủ):
  python run_pipeline.py \\
    --sketch_dir      ./sketches \\
    --photo_dir       ./photos \\
    --work_dir        ./pipeline_output \\
    --enhance         both \\
    --out_size        256 \\
    --unsharp_alpha   1.5 \\
    --dilate_kernel   2 \\
    --aug_per_image   3 \\
    --flip_prob       0.5 \\
    --rotation_deg    15 \\
    --blur_sigma      1.5 \\
    --noise_std       10 \\
    --jitter_strength 0.1 \\
    --train_ratio     0.8 \\
    --seed            42 \\
    --skip_preprocess \\
    --skip_augment
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, step_name: str):
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n[ERR] '{step_name}' thất bại (exit code {result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline CUFS: Preprocess → Augment → Combine + Train/Test split"
    )

    # ── Đường dẫn ────────────────────────────────────────────────────────────
    parser.add_argument("--sketch_dir", type=str, required=True,
                        help="Thư mục sketches gốc (flat)")
    parser.add_argument("--photo_dir",  type=str, required=True,
                        help="Thư mục photos gốc (flat)")
    parser.add_argument("--work_dir",   type=str, default="./pipeline_output",
                        help="Thư mục chứa output trung gian (mặc định: ./pipeline_output)")

    # ── Preprocess ───────────────────────────────────────────────────────────
    parser.add_argument("--enhance",       type=str,   default="both",
                        choices=["dilate", "unsharp", "both", "none"],
                        help="Edge enhancement áp lên PHOTO (mặc định: both)")
    parser.add_argument("--out_size",      type=int,   default=256,
                        help="Kích thước ảnh output (mặc định: 256)")
    parser.add_argument("--unsharp_alpha", type=float, default=1.5,
                        help="Alpha unsharp mask (mặc định: 1.5)")
    parser.add_argument("--dilate_kernel", type=int,   default=2,
                        help="Kernel size dilate (mặc định: 2)")

    # ── Augmentation ─────────────────────────────────────────────────────────
    parser.add_argument("--aug_per_image",   type=int,   default=3,
                        help="Số cặp sinh ra mỗi ảnh gốc, gồm aug0=bản gốc (mặc định: 3)")
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

    # ── Combine + Split ───────────────────────────────────────────────────────
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Tỷ lệ train (mặc định: 0.8 → 80/20)")

    # ── Chung ─────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",            type=int,  default=42,
                        help="Random seed (mặc định: 42)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Bỏ qua bước preprocess nếu đã chạy rồi")
    parser.add_argument("--skip_augment",    action="store_true",
                        help="Bỏ qua bước augmentation nếu đã chạy rồi")

    args   = parser.parse_args()
    work   = Path(args.work_dir)
    python = sys.executable

    preprocessed = work / "preprocessed"
    augmented    = work / "augmented"
    final_output = work / "output"

    # ── Bước 1: Preprocess ────────────────────────────────────────────────────
    if not args.skip_preprocess:
        run([
            python, "preprocess.py",
            "--sketch_dir",    args.sketch_dir,
            "--photo_dir",     args.photo_dir,
            "--output_dir",    str(preprocessed),
            "--enhance",       args.enhance,
            "--out_size",      str(args.out_size),
            "--unsharp_alpha", str(args.unsharp_alpha),
            "--dilate_kernel", str(args.dilate_kernel),
        ], "Bước 1/3 — Preprocess (resize photo + edge enhancement + threshold square crop sketch)")
    else:
        print("\n[SKIP] Bước 1 — Preprocess")

    # ── Bước 2: Augmentation ──────────────────────────────────────────────────
    if not args.skip_augment:
        run([
            python, "augmentation.py",
            "--sketch_dir",      str(preprocessed / "sketch"),
            "--photo_dir",       str(preprocessed / "photo"),
            "--output_dir",      str(augmented),
            "--aug_per_image",   str(args.aug_per_image),
            "--flip_prob",       str(args.flip_prob),
            "--rotation_deg",    str(args.rotation_deg),
            "--blur_sigma",      str(args.blur_sigma),
            "--noise_std",       str(args.noise_std),
            "--jitter_strength", str(args.jitter_strength),
            "--seed",            str(args.seed),
        ], "Bước 2/3 — Augmentation (flip + rotation + blur + noise + jitter)")
    else:
        print("\n[SKIP] Bước 2 — Augmentation")

    # ── Bước 3: Combine + Split ───────────────────────────────────────────────
    if args.skip_augment:
        src_sketch = str(preprocessed / "sketch")
        src_photo  = str(preprocessed / "photo")
    else:
        src_sketch = str(augmented / "sketch")
        src_photo  = str(augmented / "photo")

    run([
        python, "match_sketch_image.py",
        "--sketch_dir",  src_sketch,
        "--photo_dir",   src_photo,
        "--output_dir",  str(final_output),
        "--train_ratio", str(args.train_ratio),
        "--seed",        str(args.seed),
    ], "Bước 3/3 — Combine A|B + Train/Test split")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline hoàn tất!")
    print(f"{'='*60}")
    print(f"  Train  : {final_output / 'train'}")
    print(f"  Test   : {final_output / 'test'}")
    print(f"  Tỷ lệ  : {args.train_ratio:.0%} / {1-args.train_ratio:.0%}")
    print(f"  Seed   : {args.seed}")


if __name__ == "__main__":
    main()
