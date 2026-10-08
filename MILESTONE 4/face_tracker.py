"""
face_tracker.py - Lightweight Multi-Face Tracker for Edge AI (Raspberry Pi 4B)
Sử dụng thuật toán IOU + Centroid Matching để bám vết khuôn mặt giữa các frame.
Đặc điểm:
  - Cực nhẹ: Thời gian tính toán < 0.05ms mỗi frame trên CPU Pi 4.
  - Phân tách Nhận diện (Recognition) khỏi Bám vết (Tracking):
    -> Chỉ chạy mô hình nhận diện SFace 1-2 lần khi có mặt mới.
    -> Các frame tiếp theo chỉ cần bám vết, giúp đạt FPS > 25 - 30 ổn định.
"""

import time
import numpy as np


def compute_iou(boxA, boxB):
    # box: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class TrackedFace:
    def __init__(self, track_id, bbox, landmarks, raw_face):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.landmarks = landmarks
        self.raw_face = raw_face
        
        self.name = "Đang nhận diện..."
        self.confidence = 0.0
        self.recognized = False
        self.recognition_count = 0
        self.max_recognition_attempts = 3  # Thử nhận diện tối đa 3 frame đầu để lấy độ tin cậy cao nhất
        
        self.missed_frames = 0
        self.created_time = time.time()
        self.last_seen_time = time.time()
        self.recheck_counter = 0  # Định kỳ re-check sau mỗi N frame nếu cần

    def update_detection(self, bbox, landmarks, raw_face):
        self.bbox = bbox
        self.landmarks = landmarks
        self.raw_face = raw_face
        self.missed_frames = 0
        self.last_seen_time = time.time()
        self.recheck_counter += 1


class FaceTracker:
    def __init__(self, iou_threshold=0.3, max_missed=8, recheck_interval=60):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.recheck_interval = recheck_interval
        self.next_track_id = 1
        self.tracks = {}  # {track_id: TrackedFace}
        self.last_recognition_ms = 0.0
        self.last_recognition_count = 0

    def get_active_tracks(self):
        """Return cached tracks on frames where face detection is skipped."""
        return list(self.tracks.values())

    def update(self, detections, frame, face_engine):
        """
        Cập nhật danh sách tracks với các detections mới.
        detections: danh sách từ face_engine.detect()
        Trả về danh sách các track hiện tại để vẽ lên màn hình.
        """
        recognition_started = None
        self.last_recognition_count = 0
        matched_track_ids = set()
        unmatched_detections = []

        if len(self.tracks) > 0 and len(detections) > 0:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)

            for d_idx, det in enumerate(detections):
                for t_idx, t_id in enumerate(track_ids):
                    iou_matrix[d_idx, t_idx] = compute_iou(det['bbox'], self.tracks[t_id].bbox)

            # Match tham lam theo IOU cao nhất
            for d_idx in range(len(detections)):
                best_t_idx = int(np.argmax(iou_matrix[d_idx]))
                best_iou = iou_matrix[d_idx, best_t_idx]

                if best_iou >= self.iou_threshold:
                    t_id = track_ids[best_t_idx]
                    if t_id not in matched_track_ids:
                        det = detections[d_idx]
                        self.tracks[t_id].update_detection(det['bbox'], det['landmarks'], det['raw'])
                        matched_track_ids.add(t_id)
                        iou_matrix[:, best_t_idx] = -1.0
                    else:
                        unmatched_detections.append(detections[d_idx])
                else:
                    unmatched_detections.append(detections[d_idx])
        else:
            unmatched_detections = detections

        # Xử lý các track không tìm thấy detection tương ứng
        dead_tracks = []
        for t_id, track in self.tracks.items():
            if t_id not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    dead_tracks.append(t_id)

        for t_id in dead_tracks:
            del self.tracks[t_id]

        # Tạo track mới cho các detection chưa match
        for det in unmatched_detections:
            t_id = self.next_track_id
            self.next_track_id += 1
            new_track = TrackedFace(t_id, det['bbox'], det['landmarks'], det['raw'])
            self.tracks[t_id] = new_track

        # Tiến hành Nhận diện danh tính (Chỉ chạy khi cần thiết)
        for t_id, track in self.tracks.items():
            # Điều kiện chạy SFace:
            # 1. Chưa hoàn thành nhận diện ban đầu (recognition_count < max_recognition_attempts)
            # 2. Hoặc định kỳ recheck (sau mỗi recheck_interval frames)
            need_recognition = (not track.recognized and track.recognition_count < track.max_recognition_attempts) or \
                               (track.recheck_counter >= self.recheck_interval)

            if need_recognition:
                try:
                    if recognition_started is None:
                        recognition_started = time.perf_counter()
                    feat = face_engine.extract_feature(frame, track.raw_face)
                    name, conf = face_engine.match(feat)
                    self.last_recognition_count += 1
                    
                    track.recognition_count += 1
                    if conf > track.confidence or track.recognition_count == 1:
                        track.name = name
                        track.confidence = conf

                    # Nếu điểm đủ cao hoặc đã thử hết số lần thì đánh dấu hoàn thành
                    if track.name != "unknown" or track.recognition_count >= track.max_recognition_attempts:
                        track.recognized = True
                        track.recheck_counter = 0
                except Exception as e:
                    pass

        if recognition_started is None:
            self.last_recognition_ms = 0.0
        else:
            self.last_recognition_ms = (time.perf_counter() - recognition_started) * 1000.0

        return list(self.tracks.values())
