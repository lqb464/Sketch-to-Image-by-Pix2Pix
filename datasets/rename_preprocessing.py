# đổi dạng từ f-005-01 thành F2-005-01 cho các file đầu vào số như cài đặt
from pathlib import Path

path = Path("./cufs/raw/photos")

for i in range(5, 39):
    photo_name=f"f-{i:03d}-01"
    sketch_name=f"F2-{i:03d}-01"
    old_photo_path = path / f"{photo_name}.jpg"
    new_photo_path = path / f"{sketch_name}.jpg"
    if old_photo_path.exists():
        old_photo_path.rename(new_photo_path)

for i in range(8, 63):
    photo_name=f"m-{i:03d}-01"
    sketch_name=f"M2-{i:03d}-01"
    old_photo_path = path / f"{photo_name}.jpg"
    new_photo_path = path / f"{sketch_name}.jpg"
    if old_photo_path.exists():
        old_photo_path.rename(new_photo_path)

print("Đổi tên file hoàn tất!")