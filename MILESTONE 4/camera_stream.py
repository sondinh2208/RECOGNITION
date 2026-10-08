"""
camera_stream.py - High-Performance Threaded Camera Capture
Triệt tiêu hoàn toàn độ trễ hàng đợi buffer (Buffer Queue Latency) trên Raspberry Pi 4.
Tự động hỗ trợ V4L2 (Linux/Pi 4) và DirectShow (Windows).
"""

import sys
import threading
import time
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.target_fps = fps

        self.cap = None
        self._init_camera()

        self.grabbed = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def _init_camera(self):
        is_windows = sys.platform.startswith('win')
        
        # Thử mở camera theo nền tảng
        if is_windows:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)
        else:
            # Trên Linux / Raspberry Pi: Dùng V4L2
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened() and self.src == 0:
            # Thử cổng 1 nếu cổng 0 không mở được
            print(f"[Camera] Không mở được cổng {self.src}, đang thử cổng 1...")
            self.src = 1
            if is_windows:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print(f"[Camera] CẢNH BÁO: Không thể kết nối tới camera (cổng {self.src})!")
            return

        # Cấu hình MJPEG phần cứng
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Lấy kích thước thực tế
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Camera sẵn sàng: cổng {self.src}, độ phân giải {actual_w}x{actual_h}")

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            return False

        self.running = True
        self.grabbed, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.grabbed = True
            else:
                time.sleep(0.01)
            # Nghỉ rất nhỏ để tránh chiếm 100% nhân CPU của thread capture
            time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print("[Camera] Đã giải phóng camera.")
