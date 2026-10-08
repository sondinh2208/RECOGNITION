"""
face_embedding.py - Đăng ký khuôn mặt mới tối ưu cho Raspberry Pi 4B & PC
Sử dụng:
  - YuNet Face Detector: phát hiện khuôn mặt và 5 điểm đặc trưng (Landmarks)
  - SFace Recognizer: trích xuất vector đặc trưng 128-D
  - Tự động lưu file .npy và ảnh tham chiếu .jpg vào thư mục database/
"""

import os
import sys
import time
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from face_engine import FaceEngine, imwrite_unicode
from camera_stream import ThreadedCamera

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_PATH, exist_ok=True)


def register_face():
    print("=" * 60)
    print("       ĐĂNG KÝ KHUÔN MẶT MỚI (TỐI ƯU PI 4B)")
    print("=" * 60)

    engine = FaceEngine()
    cam = ThreadedCamera(src=0, width=640, height=480, fps=30)
    if not cam.start():
        print("[Lỗi] Không thể mở camera!")
        return

    print("\n--- HƯỚNG DẪN ĐĂNG KÝ ---")
    print("  - Nhìn thẳng vào camera, giữ khuôn mặt ở giữa khung hình")
    print("  - Nhấn phím 'S' để chụp & lưu khuôn mặt")
    print("  - Nhấn phím 'Q' để thoát\n")

    captured_data = None
    window_name = "Dang ky khuon mat (YuNet + SFace)"

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            display_frame = frame.copy()

            # Phát hiện khuôn mặt
            faces = engine.detect(frame, input_size=(w, h))

            if faces:
                for f in faces:
                    x, y, fw, fh = f['bbox']
                    score = f['score']

                    # Vẽ bounding box
                    cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

                    # Vẽ 5 điểm landmark
                    for pt in f['landmarks']:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

                    cv2.putText(
                        display_frame,
                        f"Face: {score:.2f}",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # Hướng dẫn trên khung hình
            cv2.putText(
                display_frame,
                "Nhan 'S': Chup & Luu | Nhan 'Q': Thoat",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('s'), ord('S')):
                if len(faces) == 0:
                    print("⚠️ Chưa phát hiện khuôn mặt nào trong khung hình! Vui lòng thử lại.")
                elif len(faces) > 1:
                    print("⚠️ Phát hiện nhiều hơn 1 khuôn mặt! Vui lòng chỉ đứng 1 người trước camera.")
                else:
                    best_face = faces[0]
                    feat = engine.extract_feature(frame, best_face['raw'])
                    captured_data = (feat, frame.copy())
                    print("\n📸 Đã chụp khuôn mặt thành công!")
                    break

            if key in (ord('q'), ord('Q')):
                print("Đã hủy đăng ký.")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    if captured_data is not None:
        feat, full_img = captured_data
        while True:
            name = input("Nhập tên người này (VD: NguyenVanA hoặc To Manh Hung): ").strip()
            if name:
                break
            print("Tên không được để trống!")

        npy_path = os.path.join(DATABASE_PATH, f"{name}.npy")
        jpg_path = os.path.join(DATABASE_PATH, f"{name}.jpg")

        np.save(npy_path, feat)
        imwrite_unicode(jpg_path, full_img)

        print(f"✅ Đã lưu vector SFace 128-D: {npy_path}")
        print(f"✅ Đã lưu ảnh tham chiếu:    {jpg_path}")
        input("\nNhấn Enter để quay lại menu...")


if __name__ == "__main__":
    register_face()