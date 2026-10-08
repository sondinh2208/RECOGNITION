import os
import sys

# KHÓA SỐ LUỒNG CPU NGAY TỪ ĐẦU ĐỂ TRIỆT TIỆU MỨC 100% CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import cv2
import numpy as np
import threading
import time
import warnings

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from insightface.app import FaceAnalysis

print("Đang nạp mô hình AI lên Intel Iris Xe GPU...")

# KHỞI TẠO OPENVINO ÉP ĐẨY TÍNH TOÁN SANG INTEL XE GPU (FP16)
try:
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
        provider_options=[{
            'device_type': 'GPU.0',
            'precision': 'FP16'
        }]
    )
    print("-> Đã kích hoạt Intel Xe GPU (OpenVINO - FP16) thành công!")
except Exception as e:
    print("-> Lỗi kết nối GPU, chuyển sang CPU:", e)
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )

app.prepare(ctx_id=0, det_size=(160, 160))
print("Nạp mô hình xong!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_PATH, exist_ok=True)

# Khởi tạo Camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Không mở được camera!")
    sys.exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lock = threading.Lock()
latest_frame = None
cached_faces = []
running = True

# LUỒNG AI NGẦM (GIẢM NHỊP CHẠY ĐỂ TIỆN NGHỈ CPU)
def ai_worker():
    global latest_frame, cached_faces, running
    while running:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is not None:
            small_frame = cv2.resize(frame_to_process, (160, 120))
            faces = app.get(small_frame)
            
            with lock:
                cached_faces = faces

        time.sleep(0.15)  # Nghỉ 0.15s cho AI (đủ ~6-7 FPS cho AI)

ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

print("\n--- HƯỚNG DẪN ĐĂNG KÝ ---")
print("  - Nhìn thẳng vào camera")
print("  - Nhấn phím 'S' để chụp & lưu khuôn mặt")
print("  - Nhấn phím 'Q' để thoát mà không lưu\n")

# LUỒNG HIỂN THỊ CAMERA CHÍNH
while True:
    ret, frame = cam.read()
    if not ret:
        print("Không lấy được hình ảnh từ camera!")
        break

    # GIẢM TẢI CPU CHO VÒNG LẮP VÔ TẬN
    time.sleep(0.01)

    with lock:
        latest_frame = frame.copy()
        current_faces = list(cached_faces)

    h, w = frame.shape[:2]
    scale_x = w / 160.0
    scale_y = h / 120.0

    for face in current_faces:
        raw_bbox = face.bbox.astype(float).flatten()
        x1 = int(raw_bbox[0] * scale_x)
        y1 = int(raw_bbox[1] * scale_y)
        x2 = int(raw_bbox[2] * scale_x)
        y2 = int(raw_bbox[3] * scale_y)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Face Detected",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "Nhan S: Luu khuon mat | Nhan Q: Thoat",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Dang ky khuon mat moi", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('s'), ord('S')):
        with lock:
            faces_to_save = list(cached_faces)

        if len(faces_to_save) > 0:
            print("\nĐã chụp được khuôn mặt!")
            embedding = faces_to_save[0].embedding

            running = False
            cam.release()
            cv2.destroyAllWindows()

            while True:
                name = input("Nhập tên người này (ví dụ: NguyenVanA): ").strip()
                if name:
                    break
                print("Tên không được để trống! Vui lòng nhập lại.")

            file_path = os.path.join(DATABASE_PATH, f"{name}.npy")
            np.save(file_path, embedding)
            print(f"Đã lưu khuôn mặt thành công: {file_path}")
            input("\nNhấn Enter để kết thúc...")
            break
        else:
            print("Chưa phát hiện khuôn mặt trong khung hình! Vui lòng thử lại.")

    if key in (ord('q'), ord('Q')):
        running = False
        break

if cam.isOpened():
    cam.release()
cv2.destroyAllWindows()