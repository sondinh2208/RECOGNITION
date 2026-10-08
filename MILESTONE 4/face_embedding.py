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

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import cv2
import numpy as np

cv2.setUseOptimized(True)
cv2.setNumThreads(2)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from face_engine import FaceEngine, imwrite_unicode
from camera_stream import ThreadedCamera
from latest_frame_worker import LatestFrameWorker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_PATH, exist_ok=True)


def register_face():
    print("=" * 60)
    print("       ĐĂNG KÝ KHUÔN MẶT MỚI (TỐI ƯU PI 4B)")
    print("=" * 60)

    engine = FaceEngine()
    cam = ThreadedCamera(src=0, width=640, height=480, fps=20)
    if not cam.start():
        print("[Lỗi] Không thể mở camera!")
        return

    print("\n--- HƯỚNG DẪN ĐĂNG KÝ ---")
    print("  - Nhìn thẳng vào camera, giữ khuôn mặt ở giữa khung hình")
    print("  - Nhấn phím 'S' để chụp & lưu khuôn mặt")
    print("  - Nhấn phím 'Q' để thoát\n")

    captured_data = None
    window_name = "Dang ky khuon mat (YuNet + SFace)"
    last_sequence = -1
    last_result_sequence = -1
    latest_faces = []
    latest_detection_frame = None
    display_fps = 0.0
    camera_fps = 0.0
    ai_fps = 0.0
    ai_ms = 0.0
    stats_started = None
    stats_start_sequence = None
    displayed_in_window = 0
    ai_completed_in_window = 0
    register_detect_width = max(
        128, min(320, int(os.environ.get("PI_REGISTER_DETECT_WIDTH", "192")))
    )
    register_detect_size = (
        register_detect_width,
        register_detect_width * 3 // 4,
    )

    def process_registration_frame(inference_frame):
        return {
            "faces": engine.detect_scaled(
                inference_frame, input_size=register_detect_size
            ),
            "frame": inference_frame,
        }

    inference_worker = LatestFrameWorker(
        process_registration_frame, name="registration-inference"
    )

    try:
        while True:
            ret, frame, sequence, captured_at = cam.read_new(last_sequence, timeout=0.1)
            if not ret or frame is None:
                continue

            last_sequence = sequence
            displayed_in_window += 1
            now = time.perf_counter()
            if stats_started is None:
                stats_started = now
                stats_start_sequence = sequence

            inference_worker.submit(frame, sequence, captured_at)
            ai_result = inference_worker.get_latest()
            if (
                ai_result is not None
                and ai_result["sequence"] != last_result_sequence
                and ai_result["error"] is None
            ):
                last_result_sequence = ai_result["sequence"]
                ai_completed_in_window += 1
                latest_faces = ai_result["value"]["faces"]
                latest_detection_frame = ai_result["value"]["frame"]
                ai_ms = (
                    ai_result["process_ms"] if ai_ms == 0.0
                    else ai_ms * 0.8 + ai_result["process_ms"] * 0.2
                )

            elapsed = now - stats_started
            if elapsed >= 1.0:
                display_fps = displayed_in_window / elapsed
                camera_fps = (sequence - stats_start_sequence) / elapsed
                ai_fps = ai_completed_in_window / elapsed
                stats_started = now
                stats_start_sequence = sequence
                displayed_in_window = 0
                ai_completed_in_window = 0

            display_frame = frame.copy()

            if latest_faces:
                for f in latest_faces:
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
                f"FPS:{display_fps:.1f} CAM:{camera_fps:.1f} "
                f"AI:{ai_fps:.1f} {register_detect_size[0]}x{register_detect_size[1]} "
                f"({ai_ms:.0f}ms)",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )
            cv2.putText(
                display_frame,
                "Nhan 'S': Chup & Luu | Nhan 'Q': Thoat",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                1
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('s'), ord('S')):
                if len(latest_faces) == 0 or latest_detection_frame is None:
                    print("⚠️ Chưa phát hiện khuôn mặt nào trong khung hình! Vui lòng thử lại.")
                elif len(latest_faces) > 1:
                    print("⚠️ Phát hiện nhiều hơn 1 khuôn mặt! Vui lòng chỉ đứng 1 người trước camera.")
                else:
                    inference_worker.stop()
                    best_face = latest_faces[0]
                    feat = engine.extract_feature(
                        latest_detection_frame, best_face['raw']
                    )
                    captured_data = (feat, latest_detection_frame.copy())
                    print("\n📸 Đã chụp khuôn mặt thành công!")
                    break

            if key in (ord('q'), ord('Q')):
                print("Đã hủy đăng ký.")
                break

    finally:
        inference_worker.stop()
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
