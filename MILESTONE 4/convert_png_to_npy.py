import cv2
import numpy as np
import os
import warnings
import sys

# Ẩn các cảnh báo log
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from insightface.app import FaceAnalysis

print("Đang nạp mô hình AI...")

# Khởi tạo InsightFace
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(256, 256)
)

print("Nạp mô hình xong!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")

# Lấy danh sách tất cả file PNG trong thư mục database
png_files = sorted([f for f in os.listdir(DATABASE_PATH) if f.endswith('.png')])
print(f"Tìm thấy {len(png_files)} file PNG cần chuyển đổi")

success_count = 0
error_count = 0
deleted_count = 0

for png_file in png_files:
    png_path = os.path.join(DATABASE_PATH, png_file)
    npy_file = png_file.replace('.png', '.npy')
    npy_path = os.path.join(DATABASE_PATH, npy_file)
    
    # Bỏ qua nếu file .npy đã tồn tại
    if os.path.exists(npy_path):
        print(f"  [DA CO] {png_file} - đã có file {npy_file}")
        continue
    
    try:
        # Đọc ảnh
        img = cv2.imread(png_path)
        if img is None:
            raise Exception(f"Không thể đọc file ảnh {png_file}")
        
        # Phát hiện khuôn mặt
        faces = app.get(img)
        
        if len(faces) == 0:
            raise Exception(f"Không phát hiện khuôn mặt trong {png_file}")
        
        # Lưu embedding khuôn mặt
        embedding = faces[0].embedding
        np.save(npy_path, embedding)
        print(f"  [OK] {png_file} -> {npy_file}")
        success_count += 1
        
    except Exception as e:
        print(f"  [LOI] {png_file}: {e}")
        error_count += 1
        # Xóa file PNG lỗi
        try:
            os.remove(png_path)
            print(f"  [DA XOA] Đã xóa file {png_file}")
            deleted_count += 1
        except Exception as del_err:
            print(f"  [LOI XOA] Khong the xoa file {png_file}: {del_err}")

print(f"\n--- KET QUA ---")
print(f"Thanh cong: {success_count} file")
print(f"Loi: {error_count} file")
print(f"Da xoa: {deleted_count} file")