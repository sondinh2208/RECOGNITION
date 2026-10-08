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

# Avoid oversubscribing the four Pi 4 cores from OpenCV/BLAS at the same time.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import cv2

cv2.setUseOptimized(True)
cv2.setNumThreads(2)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from face_engine import FaceEngine
from face_tracker import FaceTracker
from camera_stream import ThreadedCamera
from latest_frame_worker import LatestFrameWorker
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
    cam = ThreadedCamera(src=0, width=640, height=480, fps=20)
    if not cam.start():
        print("[Lỗi] Không thể mở camera! Vui lòng kiểm tra cổng cắm.")
        return

    print("\n[Main] Đang khởi động luồng nhận diện...")
    print("       Nhấn 'Q' tại cửa sổ hình ảnh để thoát chương trình.\n")

    window_name = "Nhan dien khuon mat (YuNet + SFace) - Pi 4B"
    last_sequence = -1
    stats_started = None
    stats_start_sequence = None
    processed_in_window = 0
    camera_fps = 0.0
    processing_fps = 0.0
    ai_fps = 0.0
    ai_completed_in_window = 0
    last_ai_sequence = -1
    detect_ms_smooth = 0.0
    process_ms_smooth = 0.0
    recognition_ms_smooth = 0.0
    last_status_print = time.monotonic()
    headless = False

    active_tracks = []
    ai_frame_age_ms = 0.0

    def process_inference(inference_frame):
        detect_started = time.perf_counter()
        detections = engine.detect_scaled(inference_frame, input_size=(320, 240))
        detect_ms = (time.perf_counter() - detect_started) * 1000.0
        tracks = tracker.update(detections, inference_frame, engine)
        snapshots = [
            {
                "bbox": tuple(track.bbox),
                "name": track.name,
                "confidence": track.confidence,
                "recognized": track.recognized,
            }
            for track in tracks
        ]
        return {
            "tracks": snapshots,
            "detect_ms": detect_ms,
            "recognition_ms": tracker.last_recognition_ms,
        }

    inference_worker = LatestFrameWorker(process_inference, name="face-inference")

    try:
        while True:
            ret, frame, sequence, captured_at = cam.read_new(last_sequence, timeout=0.1)
            if not ret or frame is None:
                continue

            last_sequence = sequence
            frame_start = time.perf_counter()
            processed_in_window += 1
            if stats_started is None:
                stats_started = frame_start
                stats_start_sequence = sequence

            # The worker owns this unmodified frame. Drawing happens on a copy below.
            inference_worker.submit(frame, sequence, captured_at)
            ai_result = inference_worker.get_latest()
            if (
                ai_result is not None
                and ai_result["sequence"] != last_ai_sequence
                and ai_result["error"] is None
            ):
                last_ai_sequence = ai_result["sequence"]
                ai_completed_in_window += 1
                result_value = ai_result["value"]
                active_tracks = result_value["tracks"]
                detect_ms = result_value["detect_ms"]
                detect_ms_smooth = (
                    detect_ms if detect_ms_smooth == 0.0
                    else detect_ms_smooth * 0.8 + detect_ms * 0.2
                )
                recognition_ms = result_value["recognition_ms"]
                if recognition_ms > 0.0:
                    recognition_ms_smooth = (
                        recognition_ms if recognition_ms_smooth == 0.0
                        else recognition_ms_smooth * 0.8 + recognition_ms * 0.2
                    )
                process_ms = ai_result["process_ms"]
                process_ms_smooth = (
                    process_ms if process_ms_smooth == 0.0
                    else process_ms_smooth * 0.85 + process_ms * 0.15
                )
                ai_frame_age_ms = max(
                    0.0, (time.perf_counter() - ai_result["captured_at"]) * 1000.0
                )

            # 5. Xử lý mở cửa khi nhận diện đúng người
            for track in active_tracks:
                if track["recognized"] and track["name"] not in ("unknown", "Đang nhận diện..."):
                    # Gửi lệnh mở cửa ESP32
                    serial_mgr.send_open(trigger_name=track["name"])
            current_time = time.perf_counter()
            stats_elapsed = current_time - stats_started
            if stats_elapsed >= 1.0:
                camera_fps = (sequence - stats_start_sequence) / stats_elapsed
                processing_fps = processed_in_window / stats_elapsed
                ai_fps = ai_completed_in_window / stats_elapsed
                stats_started = current_time
                stats_start_sequence = sequence
                processed_in_window = 0
                ai_completed_in_window = 0
            # 7. Vẽ giao diện và Bounding Box
            if not headless:
                display_frame = frame.copy()

                # Vẽ thông tin từng khuôn mặt
                for track in active_tracks:
                    x, y, bw, bh = track["bbox"]
                    name = track["name"]
                    conf = track["confidence"]

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
                info_text = (
                    f"FPS: {processing_fps:.1f} CAM:{camera_fps:.1f} | "
                    f"AI:{ai_fps:.1f} Mat:{len(active_tracks)} "
                    f"ESP32:{'OK' if serial_mgr.ser else 'OFF'}"
                )
                cv2.putText(
                    display_frame,
                    info_text,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2
                )
                timing_text = (
                    f"DET:{detect_ms_smooth:.0f}ms "
                    f"REC:{recognition_ms_smooth:.0f}ms "
                    f"PROC:{process_ms_smooth:.0f}ms AIAGE:{ai_frame_age_ms:.0f}ms"
                )
                cv2.putText(
                    display_frame,
                    timing_text,
                    (10, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    1
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
                # In trạng thái mỗi 3 giây
                if time.monotonic() - last_status_print >= 3.0:
                    print(
                        f"[Headless] FPS={processing_fps:.1f}, CAM={camera_fps:.1f}, "
                        f"AI={ai_fps:.1f}, DET={detect_ms_smooth:.1f}ms, "
                        f"REC={recognition_ms_smooth:.1f}ms, "
                        f"AIAGE={ai_frame_age_ms:.1f}ms, faces={len(active_tracks)}"
                    )
                    last_status_print = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Main] Đã nhận tín hiệu dừng từ bàn phím (Ctrl+C).")

    finally:
        inference_worker.stop()
        cam.release()
        serial_mgr.close()
        cv2.destroyAllWindows()
        print("[Main] Chương trình kết thúc an toàn.")


if __name__ == "__main__":
    run_recognition()
