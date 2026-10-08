"""Measure camera throughput and YuNet latency on the actual Raspberry Pi."""

import statistics
import time

import cv2

from camera_stream import ThreadedCamera
from face_engine import FaceEngine


def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    camera = ThreadedCamera(src=0, width=640, height=480, fps=20)
    if not camera.start():
        raise SystemExit("Cannot start camera")

    last_sequence = -1
    latest_frame = None
    frame_count = 0
    started_at = time.perf_counter()
    elapsed = 0.0
    try:
        while time.perf_counter() - started_at < 5.0:
            ok, frame, sequence, _ = camera.read_new(last_sequence, timeout=0.2)
            if not ok:
                continue
            last_sequence = sequence
            latest_frame = frame
            frame_count += 1
    finally:
        elapsed = time.perf_counter() - started_at
        camera.release()

    print(f"CAMERA_BACKEND={camera.backend_name}")
    print(f"CAMERA_FPS={frame_count / elapsed:.2f}")

    if latest_frame is None:
        raise SystemExit("Camera returned no frame")

    engine = FaceEngine(top_k=100)
    for input_size in ((160, 120), (192, 144), (320, 240)):
        samples = []
        for _ in range(25):
            sample_started = time.perf_counter()
            engine.detect_scaled(latest_frame, input_size=input_size)
            samples.append((time.perf_counter() - sample_started) * 1000.0)
        samples.sort()
        p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
        print(
            f"YUNET_{input_size[0]}x{input_size[1]}_MS="
            f"mean:{statistics.mean(samples):.2f},median:{statistics.median(samples):.2f},"
            f"p95:{p95:.2f}"
        )


if __name__ == "__main__":
    main()
