"""Cheap optical-flow projection between neural face detections."""

import copy

import cv2
import numpy as np


class OpticalFlowProjector:
    """Move detected face boxes between slower AI updates using landmarks."""

    def __init__(self, width=320, height=240):
        self.flow_size = (width, height)
        self.previous_gray = None
        self.tracks = []

    def reset(self, tracks, reference_frame):
        self.tracks = copy.deepcopy(tracks)
        self.previous_gray = self._to_gray(reference_frame)

    def update(self, frame):
        current_gray = self._to_gray(frame)
        if self.previous_gray is None:
            self.previous_gray = current_gray
            return copy.deepcopy(self.tracks)
        if not self.tracks:
            self.previous_gray = current_gray
            return []

        frame_h, frame_w = frame.shape[:2]
        scale_x = self.flow_size[0] / float(frame_w)
        scale_y = self.flow_size[1] / float(frame_h)
        points = []
        spans = []
        for track in self.tracks:
            start = len(points)
            for x, y in track.get("landmarks", []):
                points.append((x * scale_x, y * scale_y))
            spans.append((start, len(points)))

        if not points:
            self.previous_gray = current_gray
            return copy.deepcopy(self.tracks)

        old_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            old_points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if new_points is None or status is None:
            self.previous_gray = current_gray
            return copy.deepcopy(self.tracks)

        old_points = old_points.reshape(-1, 2)
        new_points = new_points.reshape(-1, 2)
        status = status.reshape(-1).astype(bool)

        for track, (start, end) in zip(self.tracks, spans):
            if end <= start:
                continue
            valid = status[start:end]
            if int(valid.sum()) < 2:
                continue

            movement = new_points[start:end][valid] - old_points[start:end][valid]
            dx_small, dy_small = np.median(movement, axis=0)
            dx = float(dx_small / scale_x)
            dy = float(dy_small / scale_y)

            # Reject broken optical flow rather than jumping a box across frame.
            if abs(dx) > frame_w * 0.2 or abs(dy) > frame_h * 0.2:
                continue

            x, y, box_w, box_h = track["bbox"]
            x = max(0, min(frame_w - box_w, int(round(x + dx))))
            y = max(0, min(frame_h - box_h, int(round(y + dy))))
            track["bbox"] = (x, y, box_w, box_h)

            translated = []
            for point_index, (old_x, old_y) in enumerate(track.get("landmarks", [])):
                point_offset = start + point_index
                if point_offset < end and status[point_offset]:
                    next_x = new_points[point_offset][0] / scale_x
                    next_y = new_points[point_offset][1] / scale_y
                    translated.append((float(next_x), float(next_y)))
                else:
                    translated.append((old_x + dx, old_y + dy))
            track["landmarks"] = translated

        self.previous_gray = current_gray
        return copy.deepcopy(self.tracks)

    def _to_gray(self, frame):
        resized = cv2.resize(frame, self.flow_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
