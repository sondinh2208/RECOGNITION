import os
import sys

# KHÓA SỐ LUỒNG CPU NGAY TỪ ĐẦU ĐỂ TRIỆT TIỆU MỨC 100% CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import cv2
import numpy as np
import serial
import time
import threading
import warnings

warnings.filterwarnings('ignore')

from insightface.app import FaceAnalysis

# --- 1. CẤU HÌNH SERIAL ESP32 ---
SERIAL_PORT = 'COM6'
BAUD_RATE = 9600
ser = None

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Đã kết nối tới ESP32 qua {SERIAL_PORT}")
except Exception as e:
    print(f"Không kết nối được ESP32: {e}")
    print("Chương trình sẽ chạy bình thường (không điều khiển servo)")

# --- 2. KHỞI TẠO AI ÉP CHẠY TRÊN INTEL IRIS XE GPU ---
print("Đang nạp mô hình AI lên Intel Iris Xe GPU...")
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

# --- 3. LOAD DATABASE KHUÔN MẶT ---
database = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_PATH, exist_ok=True)

for file in os.listdir(DATABASE_PATH):
    if file.endswith(".npy"):
        name, _ = os.path.splitext(file)
        vector = np.load(os.path.join(DATABASE_PATH, file))
        database[name] = vector

print("Danh sách người trong Database:", list(database.keys()))

def cosine(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# --- 4. CẤU HÌNH CAMERA ---
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Không mở được camera!")
    sys.exit()

cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)

last_open_time = 0
SERIAL_COOLDOWN = 3.0

lock = threading.Lock()
latest_frame = None
cached_results = []
running = True

# --- 5. LUỒNG XỬ LÝ AI CHẠY NGẦM ---
def ai_worker():
    global latest_frame, cached_results, last_open_time, running

    while running:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is not None:
            h, w = frame_to_process.shape[:2]
            scale_x = w / 160.0
            scale_y = h / 120.0
            small_frame = cv2.resize(frame_to_process, (160, 120))

            faces = app.get(small_frame)
            new_results = []
            current_time = time.time()

            for face in faces:
                current = face.embedding
                best_name = "unknown"
                best_conf = 0.0

                for name, vector in database.items():
                    conf = cosine(current, vector)
                    if conf > best_conf:
                        best_conf = conf
                        best_name = name

                if best_conf < 0.7:
                    best_name = "unknown"

                # Gửi lệnh mở cửa ESP32
                if best_name != "unknown" and ser is not None:
                    if current_time - last_open_time > SERIAL_COOLDOWN:
                        try:
                            ser.write(b"OPEN\n")
                            last_open_time = current_time
                            print(f"Đã gửi lệnh OPEN tới ESP32 ({best_name})")
                        except Exception as e:
                            print(f"Lỗi gửi serial: {e}")

                raw_bbox = face.bbox.astype(float).flatten()
                x1 = int(raw_bbox[0] * scale_x)
                y1 = int(raw_bbox[1] * scale_y)
                x2 = int(raw_bbox[2] * scale_x)
                y2 = int(raw_bbox[3] * scale_y)
                bbox = [x1, y1, x2, y2]
                new_results.append((bbox, best_name, best_conf))

            with lock:
                cached_results = new_results

        time.sleep(0.15)  # Tăng thời gian nghỉ cho AI worker

ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

prev_time = time.time()
fps_display = 0

# --- 6. LUỒNG HIỂN THỊ CAMERA CHÍNH ---
while True:
    ret, frame = cam.read()
    if not ret:
        print("Không lấy được frame")
        break

    # NGHỈ DỪNG ĐỂ GIẢM TẢI CPU
    time.sleep(0.01)

    with lock:
        latest_frame = frame.copy()
        draw_results = list(cached_results)

    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-6)
    prev_time = current_time
    fps_display = int(fps_display * 0.8 + fps * 0.2)

    for bbox, best_name, best_conf in draw_results:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        color = (0, 255, 0) if best_name != "unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{best_name}:{best_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.putText(
        frame,
        f"FPS: {fps_display}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        running = False
        break

cam.release()
cv2.destroyAllWindows()