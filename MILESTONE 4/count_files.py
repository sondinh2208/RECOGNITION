import os
import glob

database_path = "database"
png_files = glob.glob(os.path.join(database_path, "*.png"))
npy_files = glob.glob(os.path.join(database_path, "*.npy"))

print(f"So file PNG: {len(png_files)}")
print(f"So file NPY: {len(npy_files)}")

# Liệt kê file PNG chưa có NPY
for png in sorted(png_files):
    npy = png.replace('.png', '.npy')
    if not os.path.exists(npy):
        print(f"Chua co NPY: {os.path.basename(png)}")