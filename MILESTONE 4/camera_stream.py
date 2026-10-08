"""Low-latency threaded camera capture for Raspberry Pi and Windows."""

import os
import sys
import threading
import time

import cv2


class ThreadedCamera:
    """Keep only the newest camera frame.

    On Raspberry Pi, Picamera2 is preferred for a CSI camera. If Picamera2 is
    unavailable (for example with a USB webcam), the class falls back to V4L2.
    Set PI_CAMERA_BACKEND=v4l2 or picamera2 to force a backend.
    """

    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.target_fps = fps

        self.cap = None
        self.picam2 = None
        self.backend_name = "uninitialized"

        self.grabbed = False
        self.frame = None
        self.frame_sequence = 0
        self.frame_captured_at = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.thread = None

        self._init_camera()

    def _init_camera(self):
        is_windows = sys.platform.startswith("win")
        preference = os.environ.get("PI_CAMERA_BACKEND", "auto").lower()

        if not is_windows and preference in ("auto", "picamera2"):
            if self._init_picamera2():
                return
            if preference == "picamera2":
                print("[Camera] Picamera2 unavailable; falling back to V4L2.")

        self._init_opencv(is_windows)

    def _init_picamera2(self):
        picam2 = None
        try:
            from picamera2 import Picamera2

            picam2 = Picamera2(camera_num=self.src)
            config_args = {
                "main": {"size": (self.width, self.height), "format": "BGR888"},
                "controls": {"FrameRate": float(self.target_fps)},
                "buffer_count": 4,
            }
            try:
                config = picam2.create_video_configuration(queue=False, **config_args)
            except TypeError:
                config = picam2.create_video_configuration(**config_args)
            picam2.configure(config)
            self.picam2 = picam2
            self.backend_name = "Picamera2"
            print(
                f"[Camera] Backend=Picamera2, CSI {self.src}, "
                f"{self.width}x{self.height}@{self.target_fps}"
            )
            return True
        except Exception as error:
            if picam2 is not None:
                try:
                    picam2.close()
                except Exception:
                    pass
            self.picam2 = None
            if os.environ.get("PI_CAMERA_BACKEND", "auto").lower() == "picamera2":
                print(f"[Camera] Picamera2 initialization failed: {error}")
            return False

    def _init_opencv(self, is_windows):
        api = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2
        self.backend_name = "DirectShow" if is_windows else "V4L2"
        self.cap = cv2.VideoCapture(self.src, api)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.src)
            self.backend_name = "OpenCV-default"

        if not self.cap.isOpened() and self.src == 0:
            print("[Camera] Port 0 unavailable; trying port 1...")
            self.src = 1
            self.cap.release()
            self.cap = cv2.VideoCapture(self.src, api)
            self.backend_name = "DirectShow" if is_windows else "V4L2"
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.src)
                self.backend_name = "OpenCV-default"

        if not self.cap.isOpened():
            print(f"[Camera] Cannot open camera port {self.src}.")
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        )
        print(
            f"[Camera] Backend={self.backend_name}, port {self.src}, "
            f"{actual_w}x{actual_h}, reported FPS={actual_fps:.1f}, "
            f"FOURCC={actual_fourcc!r}"
        )

    def start(self):
        if self.picam2 is not None:
            try:
                self.picam2.start()
            except Exception as error:
                print(f"[Camera] Cannot start Picamera2: {error}")
                return False
        elif self.cap is None or not self.cap.isOpened():
            return False

        self.running = True
        grabbed, frame = self._read_backend()
        with self.frame_ready:
            self.grabbed = grabbed
            self.frame = frame
            if grabbed and frame is not None:
                self.frame_sequence = 1
                self.frame_captured_at = time.perf_counter()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _read_backend(self):
        if self.picam2 is not None:
            try:
                frame = self.picam2.capture_array("main")
                return frame is not None, frame
            except Exception as error:
                if self.running:
                    print(f"[Camera] Picamera2 read failed: {error}")
                return False, None
        return self.cap.read()

    def _capture_loop(self):
        while self.running:
            if self.picam2 is None and (self.cap is None or not self.cap.isOpened()):
                break
            grabbed, frame = self._read_backend()
            if grabbed:
                with self.frame_ready:
                    self.frame = frame
                    self.grabbed = True
                    self.frame_sequence += 1
                    self.frame_captured_at = time.perf_counter()
                    self.frame_ready.notify_all()
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def read_new(self, last_sequence=-1, timeout=0.1):
        """Return only a newly captured frame and its monotonic timestamp."""
        with self.frame_ready:
            if self.running and self.frame_sequence == last_sequence:
                self.frame_ready.wait(timeout=timeout)

            if self.frame is None or self.frame_sequence == last_sequence:
                return False, None, self.frame_sequence, self.frame_captured_at

            return (
                self.grabbed,
                self.frame.copy(),
                self.frame_sequence,
                self.frame_captured_at,
            )

    def release(self):
        self.running = False
        with self.frame_ready:
            self.frame_ready.notify_all()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
        print(f"[Camera] Released backend {self.backend_name}.")
