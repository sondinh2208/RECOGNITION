"""
convert_database.py - Chuyển đổi dữ liệu khuôn mặt sang định dạng SFace 128-D
Tối ưu hóa và chuẩn hóa toàn bộ cơ sở dữ liệu khuôn mặt cho Raspberry Pi 4.
"""

import os
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from face_engine import FaceEngine, imread_unicode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")


def convert_all():
    print("=" * 60)
    print("   CHUYỂN ĐỔI DATABASE KHUÔN MẶT SANG SFACE 128-D CHO PI 4")
    print("=" * 60)

    engine = FaceEngine()
    if not os.path.exists(DATABASE_PATH):
        print("Thư mục database không tồn tại!")
        return

    # Lấy danh sách file ảnh
    img_files = sorted([f for f in os.listdir(DATABASE_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Tìm thấy {len(img_files)} file ảnh trong {DATABASE_PATH}")

    success_count = 0
    fail_count = 0

    for idx, img_name in enumerate(img_files, 1):
        img_path = os.path.join(DATABASE_PATH, img_name)
        base_name = os.path.splitext(img_name)[0]
        npy_path = os.path.join(DATABASE_PATH, f"{base_name}.npy")

        # Đọc ảnh an toàn hỗ trợ unicode
        img = imread_unicode(img_path)
        if img is None:
            print(f"[{idx}/{len(img_files)}] ❌ Không đọc được: {img_name}")
            fail_count += 1
            continue

        h, w = img.shape[:2]
        faces = engine.detect(img, input_size=(w, h))

        if not faces:
            print(f"[{idx}/{len(img_files)}] ⚠️ Không phát hiện khuôn mặt: {img_name}")
            fail_count += 1
            continue

        # Lấy khuôn mặt có kích thước/score lớn nhất
        best_face = max(faces, key=lambda f: f['score'])
        feat = engine.extract_feature(img, best_face['raw'])

        # Lưu file .npy chuẩn 128-d
        np.save(npy_path, feat)
        success_count += 1
        if idx % 20 == 0 or idx == len(img_files):
            print(f"[{idx}/{len(img_files)}] ✅ Đã tạo vector SFace: {base_name}.npy (shape: {feat.shape})")

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT: {success_count} thành công, {fail_count} thất bại.")
    print("=" * 60)


if __name__ == "__main__":
    convert_all()
