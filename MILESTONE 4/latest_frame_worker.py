"""Single-slot background worker for low-latency camera inference."""

import threading
import time


class LatestFrameWorker:
    """Process only the newest submitted frame on a background thread.

    The queue has one slot: if inference is slower than the camera, an old
    pending frame is replaced instead of building latency.
    """

    def __init__(self, processor, name="inference-worker"):
        self.processor = processor
        self.condition = threading.Condition()
        self.pending = None
        self.latest_result = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self.thread.start()

    def submit(self, frame, sequence, captured_at):
        with self.condition:
            self.pending = (frame, sequence, captured_at)
            self.condition.notify()

    def get_latest(self):
        with self.condition:
            return self.latest_result

    def _loop(self):
        while True:
            with self.condition:
                while self.running and self.pending is None:
                    self.condition.wait()
                if not self.running:
                    return
                frame, sequence, captured_at = self.pending
                self.pending = None

            started_at = time.perf_counter()
            try:
                value = self.processor(frame)
                error = None
            except Exception as exc:
                value = None
                error = str(exc)

            completed_at = time.perf_counter()
            result = {
                "value": value,
                "sequence": sequence,
                "captured_at": captured_at,
                "completed_at": completed_at,
                "process_ms": (completed_at - started_at) * 1000.0,
                "error": error,
            }
            with self.condition:
                self.latest_result = result

    def stop(self, timeout=2.0):
        with self.condition:
            if not self.running:
                return
            self.running = False
            self.pending = None
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
