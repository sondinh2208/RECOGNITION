import os
from datasets import load_dataset

# Tạo thư mục lưu ảnh
os.makedirs("database", exist_ok=True)

# Đọc dataset theo chế độ streaming
dataset = load_dataset(
    "RichardErkhov/OneMillionFaces",
    split="train",
    streaming=True
)

# Tải 1000 ảnh đầu tiên
for i, sample in enumerate(dataset):
    image = sample["png"]      # PIL Image
    image.save(f"database/{i:04d}.png")

    if (i + 1) % 100 == 0:
        print(f"Đã tải {i + 1} ảnh")

    if i == 999:
        break

print("Hoàn thành!")