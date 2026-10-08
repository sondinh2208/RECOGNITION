"""
recognition.py - Hệ thống nhận diện khuôn mặt thời gian thực & Mở cửa ESP32
Tối ưu hóa đặc biệt cho Raspberry Pi 4B (FPS > 20 - 25):
  1. Face Detection: OpenCV YuNet siêu nhẹ (ARM NEON)
  2. Face Recognition: SFace 128-D + Ma trận tìm kiếm NumPy (< 0.1ms)
  3. Bám vết khuôn mặt (Face Tracker): Tách biệt Recognition khỏi Tracking
  4. Thu nhận khung hình đa luồng (Threaded Camera): Triệt tiêu độ trễ hàng đợi
  5. Quản lý Serial ESP32 tự động (Tương thích cả Linux Pi 4 và Windows)
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

from face_engine import FaceEngine
from face_tracker import FaceTracker
from camera_stream import ThreadedCamera
from serial_manager import SerialManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")


def run_recognition():
    print("=" * 65)
    print("   HỆ THỐNG NHẬN DIỆN KHUÔN MẶT REAL-TIME (TỐI ƯU PI 4B)")
    print("=" * 65)

    # 1. Khởi tạo AI Engine
    engine = FaceEngine(score_threshold=0.6, cosine_threshold=0.42)
    db_count = engine.load_database(DATABASE_PATH)
    print(f"[Main] Sẵn sàng đối soát với {db_count} người trong database.")

    # 2. Khởi tạo Tracker
    tracker = FaceTracker(iou_threshold=0.3, max_missed=6, recheck_interval=90)

    # 3. Khởi tạo Kết nối ESP32
    serial_mgr = SerialManager(cooldown=4.0)

    # 4. Khởi tạo Camera đa luồng (640x480)
    cam = ThreadedCamera(src=0, width=640, height=480, fps=30)
    if not cam.start():
        print("[Lỗi] Không thể mở camera! Vui lòng kiểm tra cổng cắm.")
        return

    print("\n[Main] Đang khởi động luồng nhận diện...")
    print("       Nhấn 'Q' tại cửa sổ hình ảnh để thoát chương trình.\n")

    window_name = "Nhan dien khuon mat (YuNet + SFace) - Pi 4B"
    prev_time = time.time()
    fps_smooth = 0.0
    headless = False

    # Kích thước ảnh dùng để detect (320x240 để đạt tốc độ cực đại trên Pi 4)
    DET_WIDTH = 320
    DET_HEIGHT = 240

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_start = time.time()
            h, w = frame.shape[:2]

            # 1. Resize khung hình phục vụ phát hiện khuôn mặt nhanh
            small_frame = cv2.resize(frame, (DET_WIDTH, DET_HEIGHT), interpolation=cv2.INTER_LINEAR)
            scale_x = w / float(DET_WIDTH)
            scale_y = h / float(DET_HEIGHT)

            # 2. Chạy YuNet Face Detection trên ảnh thu nhỏ
            small_faces = engine.detect(small_frame, input_size=(DET_WIDTH, DET_HEIGHT))

            # 3. Chuyển đổi tọa độ detection về độ phân giải gốc 640x480
            scaled_detections = []
            for sf in small_faces:
                bx, by, bw, bh = sf['bbox']
                orig_bbox = [
                    int(bx * scale_x),
                    int(by * scale_y),
                    int(bw * scale_x),
                    int(bh * scale_y)
                ]

                # Map các điểm landmark và raw face về ảnh gốc
                raw = sf['raw'].copy()
                raw[0] *= scale_x
                raw[1] *= scale_y
                raw[2] *= scale_x
                raw[3] *= scale_y
                for p_idx in range(5):
                    raw[4 + p_idx * 2] *= scale_x
                    raw[5 + p_idx * 2] *= scale_y

                landmarks = []
                for pt in sf['landmarks']:
                    landmarks.append((pt[0] * scale_x, pt[1] * scale_y))

                scaled_detections.append({
                    'bbox': orig_bbox,
                    'landmarks': landmarks,
                    'score': sf['score'],
                    'raw': raw
                })

            # 4. Cập nhật Tracker (Chỉ chạy SFace nhận diện trên ảnh gốc khi có track mới)
            active_tracks = tracker.update(scaled_detections, frame, engine)

            # 5. Xử lý mở cửa khi nhận diện đúng người
            for track in active_tracks:
                if track.recognized and track.name != "unknown" and track.name != "Đang nhận diện...":
                    # Gửi lệnh mở cửa ESP32
                    serial_mgr.send_open(trigger_name=track.name)

            # 6. Tính toán FPS mượt mà
            current_time = time.time()
            instant_fps = 1.0 / max(1e-5, current_time - prev_time)
            prev_time = current_time
            fps_smooth = fps_smooth * 0.85 + instant_fps * 0.15

            # 7. Vẽ giao diện và Bounding Box
            if not headless:
                display_frame = frame.copy()

                # Vẽ thông tin từng khuôn mặt
                for track in active_tracks:
                    x, y, bw, bh = track.bbox
                    name = track.name
                    conf = track.confidence

                    if name == "unknown":
                        color = (0, 0, 255)       # Đỏ: Người lạ
                        label = f"Unknown ({conf:.2f})"
                    elif name == "Đang nhận diện...":
                        color = (0, 255, 255)     # Vàng: Đang phân tích
                        label = name
                    else:
                        color = (0, 255, 0)       # Xanh lá: Nhận diện thành công
                        label = f"{name} ({conf:.2f})"

                    # Khung bao
                    cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), color, 2)

                    # Nền nhãn tên
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = max(25, y - 8)
                    cv2.rectangle(
                        display_frame,
                        (x, label_y - label_size[1] - 4),
                        (x + label_size[0] + 6, label_y + 4),
                        color,
                        -1
                    )
                    cv2.putText(
                        display_frame,
                        label,
                        (x + 3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )

                # Vẽ thanh thông tin trạng thái trên cùng
                info_text = f"FPS: {fps_smooth:.1f} | Mat: {len(active_tracks)} | ESP32: {'OK' if serial_mgr.ser else 'OFF'}"
                cv2.putText(
                    display_frame,
                    info_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )

                try:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        print("[Main] Đã nhấn phím Q để thoát.")
                        break
                except Exception as gui_err:
                    print(f"[Main] Không có màn hình GUI ({gui_err}), chuyển sang chế độ dòng lệnh Headless.")
                    headless = True

            else:
                # Chế độ headless (không gắn màn hình, chạy qua SSH)
                time.sleep(0.01)
                # In trạng thái mỗi 3 giây
                if int(current_time) % 3 == 0 and int(prev_time) != int(current_time):
                    print(f"[Headless Status] FPS: {fps_smooth:.1f} | Đang theo dõi {len(active_tracks)} khuôn mặt")

    except KeyboardInterrupt:
        print("\n[Main] Đã nhận tín hiệu dừng từ bàn phím (Ctrl+C).")

    finally:
        cam.release()
        serial_mgr.close()
        cv2.destroyAllWindows()
        print("[Main] Chương trình kết thúc an toàn.")


if __name__ == "__main__":
    run_recognition()