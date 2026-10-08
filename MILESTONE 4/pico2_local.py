import argparse
import json
import os
import sqlite3
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
PHOTO_DIR = BASE_DIR / "pico2_photos"
SQLITE_PATH = BASE_DIR / "pico2_attendance.db"
CONFIG_PATH = BASE_DIR / "pico2_config.json"
PICO2_CLASS_DIR_NAME = "_pico2_classes"

CAMERA_INDEX = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
PROCESS_WIDTH = 160
PROCESS_HEIGHT = 120
FACE_SIZE = 48
MAX_PHOTOS = 10
DEFAULT_THRESHOLD = 0.60
REPEAT_GUARD_SECONDS = 8.0
ANOMALY_GRACE_SECONDS = 0.0


def ensure_dirs():
    DB_DIR.mkdir(exist_ok=True)
    template_root().mkdir(exist_ok=True)
    PHOTO_DIR.mkdir(exist_ok=True)


def template_root():
    return DB_DIR / PICO2_CLASS_DIR_NAME


def load_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "bus_id": "BUS_01",
        "threshold": DEFAULT_THRESHOLD,
        "alert_webhook_url": "",
        "camera_index": CAMERA_INDEX,
    }


def init_db():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            class_name TEXT,
            mode TEXT NOT NULL,
            confidence REAL NOT NULL,
            photo_path TEXT,
            created_at TEXT NOT NULL,
            day TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            sent INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def safe_name(value):
    keep = []
    for char in value.strip():
        if char.isalnum() or char in (" ", "-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip().replace("  ", " ")
    return cleaned or "unknown"


def class_dir(class_name):
    return template_root() / safe_name(class_name or "Chua nhap")


def student_file(name, class_name):
    return class_dir(class_name) / f"{safe_name(name)}.npz"


def create_class(class_name):
    class_dir(class_name).mkdir(parents=True, exist_ok=True)


def list_classes():
    root = template_root()
    if not root.exists():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir()])


def display_label(value):
    return str(value).encode("ascii", errors="ignore").decode("ascii") or "student"


def load_student_features(name, class_name):
    path = student_file(name, class_name)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["features"].astype(np.float32)


def open_camera(index):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 30)
    return cam


def load_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("Cannot load Haar face detector")
    return detector


def detect_largest_face(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
    faces = detector.detectMultiScale(
        small,
        scaleFactor=1.12,
        minNeighbors=4,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None, small

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    scale_x = frame.shape[1] / PROCESS_WIDTH
    scale_y = frame.shape[0] / PROCESS_HEIGHT
    pad = 0.16
    x1 = max(0, int((x - w * pad) * scale_x))
    y1 = max(0, int((y - h * pad) * scale_y))
    x2 = min(frame.shape[1], int((x + w * (1 + pad)) * scale_x))
    y2 = min(frame.shape[0], int((y + h * (1 + pad)) * scale_y))
    return (x1, y1, x2, y2), small


def preprocess_face(frame, bbox):
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return gray


def lbp_image(gray):
    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    code |= ((gray[:-2, :-2] >= center) << 7).astype(np.uint8)
    code |= ((gray[:-2, 1:-1] >= center) << 6).astype(np.uint8)
    code |= ((gray[:-2, 2:] >= center) << 5).astype(np.uint8)
    code |= ((gray[1:-1, 2:] >= center) << 4).astype(np.uint8)
    code |= ((gray[2:, 2:] >= center) << 3).astype(np.uint8)
    code |= ((gray[2:, 1:-1] >= center) << 2).astype(np.uint8)
    code |= ((gray[2:, :-2] >= center) << 1).astype(np.uint8)
    code |= (gray[1:-1, :-2] >= center).astype(np.uint8)
    return code


def extract_feature(face_gray):
    face_f = face_gray.astype(np.float32) / 255.0
    mean = float(face_f.mean())
    std = float(face_f.std()) + 1e-6
    normalized = (face_f - mean) / std

    tiny = cv2.resize(normalized, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)

    lbp = lbp_image(face_gray)
    lbp_features = []
    cell_h = lbp.shape[0] // 4
    cell_w = lbp.shape[1] // 4
    for row in range(4):
        for col in range(4):
            cell = lbp[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            hist = np.histogram(cell, bins=16, range=(0, 256))[0].astype(np.float32)
            hist /= hist.sum() + 1e-6
            lbp_features.append(hist)

    gx = cv2.Sobel(face_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(face_f, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    hog_features = []
    for row in range(4):
        for col in range(4):
            y1 = row * 12
            y2 = y1 + 12
            x1 = col * 12
            x2 = x1 + 12
            hist = np.histogram(
                angle[y1:y2, x1:x2],
                bins=8,
                range=(0, np.pi * 2),
                weights=mag[y1:y2, x1:x2],
            )[0].astype(np.float32)
            hist /= hist.sum() + 1e-6
            hog_features.append(hist)

    feature = np.concatenate([tiny, *lbp_features, *hog_features]).astype(np.float32)
    norm = np.linalg.norm(feature) + 1e-6
    return feature / norm


def load_students(class_name=None):
    students = {}
    root = template_root()
    if not root.exists():
        return students
    search_root = class_dir(class_name) if class_name else root
    paths = search_root.glob("*.npz") if class_name else root.glob("*/*.npz")
    for path in paths:
        data = np.load(path, allow_pickle=True)
        name = str(data["name"])
        class_name = str(data["class_name"]) if "class_name" in data else ""
        features = data["features"].astype(np.float32)
        student_id = f"{class_name}/{name}"
        students[student_id] = {"name": name, "class_name": class_name, "features": features}
    return students


def save_student(name, class_name, features):
    create_class(class_name)
    file_path = student_file(name, class_name)
    np.savez_compressed(
        file_path,
        name=name,
        class_name=class_name,
        features=np.asarray(features, dtype=np.float32),
    )


def recognize(feature, students):
    best_name = "unknown"
    best_score = -1.0
    best_class = ""
    for _student_id, info in students.items():
        templates = info["features"]
        scores = templates @ feature
        score = float(scores.max())
        if score > best_score:
            best_score = score
            best_name = info["name"]
            best_class = info["class_name"]
    return best_name, best_class, best_score


def save_rolling_photo(name, frame):
    student_dir = PHOTO_DIR / safe_name(name)
    student_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(student_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    while len(existing) >= MAX_PHOTOS:
        existing.pop(0).unlink(missing_ok=True)
    stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
    path = student_dir / f"{stamp}.jpg"
    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return str(path)


def record_attendance(conn, name, class_name, mode, confidence, photo_path):
    now = datetime.now()
    conn.execute(
        """
        INSERT INTO attendance(student_name, class_name, mode, confidence, photo_path, created_at, day)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (name, class_name, mode, confidence, photo_path, now.isoformat(timespec="seconds"), now.strftime("%Y-%m-%d")),
    )
    conn.commit()


def today_counts(conn, name, class_name=None):
    day = datetime.now().strftime("%Y-%m-%d")
    if class_name is None:
        rows = conn.execute(
            "SELECT mode, COUNT(*) FROM attendance WHERE student_name=? AND day=? GROUP BY mode",
            (name, day),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT mode, COUNT(*) FROM attendance WHERE student_name=? AND class_name=? AND day=? GROUP BY mode",
            (name, class_name, day),
        ).fetchall()
    counts = {"len": 0, "xuong": 0}
    for mode, count in rows:
        counts[mode] = count
    return counts


def detect_anomaly(conn, name, mode, class_name=None):
    counts = today_counts(conn, name, class_name)
    if mode == "xuong" and counts["xuong"] > counts["len"]:
        return "off_without_board", f"{name} has more off-board than on-board records"
    if mode == "len" and counts["len"] - counts["xuong"] > 1:
        return "double_board", f"{name} appears to board again without getting off"
    return None, None


def should_record_attendance(conn, name, class_name, mode):
    counts = today_counts(conn, name, class_name)
    if mode == "len":
        return counts["len"] <= counts["xuong"]
    if mode == "xuong":
        return counts["xuong"] < counts["len"]
    return False


def save_anomaly(conn, name, anomaly_type, message):
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO anomalies(student_name, type, message, sent, created_at) VALUES (?, ?, ?, 0, ?)",
        (name, anomaly_type, message, now),
    )
    conn.commit()


def send_anomaly_if_configured(config, name, anomaly_type, message):
    url = config.get("alert_webhook_url", "").strip()
    if not url:
        return False
    payload = {
        "bus_id": config.get("bus_id", "BUS_01"),
        "student_name": name,
        "type": anomaly_type,
        "message": message,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=4).read()
        return True
    except Exception:
        return False


def register(args):
    ensure_dirs()
    detector = load_detector()
    cam = open_camera(args.camera)
    features = []
    last_capture = 0.0

    print("Register mode")
    print("Look at camera. Press S to save one template, Q to finish.")
    print("For better accuracy, save 5-10 templates with small head movement.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            bbox, _ = detect_largest_face(detector, frame)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"templates: {len(features)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            cv2.imshow("Pico2 Register", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")) and bbox is not None and time.time() - last_capture > 0.4:
                face = preprocess_face(frame, bbox)
                if face is not None:
                    features.append(extract_feature(face))
                    last_capture = time.time()
                    print(f"Saved template {len(features)}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if not features:
        print("No template saved.")
        return
    save_student(args.name, args.class_name, features)
    print(f"Saved {len(features)} templates for {args.name}")


def run_attendance(args):
    ensure_dirs()
    config = load_config()
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", DEFAULT_THRESHOLD))
    students = load_students(args.class_name)
    if not students:
        print("No students in selected class/database. Register first.")
        return

    detector = load_detector()
    conn = init_db()
    cam = open_camera(args.camera)
    mode = args.mode
    last_seen = {}
    cached_bbox = None
    cached_result = ("unknown", "", 0.0)
    frame_id = 0
    prev_time = time.perf_counter()
    fps_smooth = 0.0

    print("Attendance mode")
    print("Keys: L=on-board, X=off-board, Q=quit")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            frame_id += 1
            now_perf = time.perf_counter()
            fps = 1.0 / max(now_perf - prev_time, 1e-6)
            prev_time = now_perf
            fps_smooth = fps if fps_smooth == 0.0 else fps_smooth * 0.86 + fps * 0.14

            should_detect = cached_bbox is None or frame_id % args.detect_every == 0
            if should_detect:
                bbox, _ = detect_largest_face(detector, frame)
                cached_bbox = bbox
            else:
                bbox = cached_bbox

            if bbox is not None:
                face = preprocess_face(frame, bbox)
                if face is not None and frame_id % args.recognize_every == 0:
                    feature = extract_feature(face)
                    name, class_name, score = recognize(feature, students)
                    if score < threshold:
                        name, class_name = "unknown", ""
                    cached_result = (name, class_name, score)

                    if name != "unknown":
                        guard_key = f"{mode}:{class_name}:{name}"
                        if time.time() - last_seen.get(guard_key, 0.0) > REPEAT_GUARD_SECONDS:
                            if not should_record_attendance(conn, name, class_name, mode):
                                continue

                            photo_path = save_rolling_photo(f"{class_name}_{name}", frame)
                            record_attendance(conn, name, class_name, mode, score, photo_path)
                            anomaly_type, message = detect_anomaly(conn, name, mode, class_name)
                            if anomaly_type:
                                save_anomaly(conn, name, anomaly_type, message)
                                send_anomaly_if_configured(config, name, anomaly_type, message)
                                print("ANOMALY:", message)
                            print(f"{mode.upper()} {class_name}/{name} {score:.2f}")
                            last_seen[guard_key] = time.time()

                x1, y1, x2, y2 = bbox
                name, _, score = cached_result
                color = (0, 220, 0) if name != "unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{display_label(cached_result[1])}/{display_label(name)}:{score:.2f}"
                cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.putText(frame, f"MODE:{mode.upper()} FPS:{fps_smooth:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)
            cv2.imshow("Pico2 Local Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("l"), ord("L")):
                mode = "len"
                print("Mode: LEN")
            elif key in (ord("x"), ord("X")):
                mode = "xuong"
                print("Mode: XUONG")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cam.release()
        conn.close()
        cv2.destroyAllWindows()


def report(_args):
    conn = init_db()
    day = datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute(
        """
        SELECT student_name,
               SUM(CASE WHEN mode='len' THEN 1 ELSE 0 END) AS board_count,
               SUM(CASE WHEN mode='xuong' THEN 1 ELSE 0 END) AS off_count
        FROM attendance
        WHERE day=?
        GROUP BY student_name
        ORDER BY student_name
        """,
        (day,),
    ).fetchall()
    print(f"Report {day}")
    if not rows:
        print("No attendance records.")
    for name, board_count, off_count in rows:
        board_count = board_count or 0
        off_count = off_count or 0
        status = "OK" if board_count == off_count else "ANOMALY"
        print(f"{name}: len={board_count} xuong={off_count} {status}")
    conn.close()


def self_test(_args):
    ensure_dirs()
    init_db().close()
    dummy = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    cv2.circle(dummy, (24, 18), 9, 180, -1)
    cv2.rectangle(dummy, (14, 30), (34, 43), 120, -1)
    feature = extract_feature(dummy)
    save_student("Self Test", "TEST", [feature])
    students = load_students()
    name, class_name, score = recognize(feature, students)
    print("SELF_TEST_NAME=", name)
    print("SELF_TEST_CLASS=", class_name)
    print("SELF_TEST_SCORE=", round(score, 4))
    print("SELF_TEST_OK=", name == "Self Test" and score > 0.99)


def parse_args():
    parser = argparse.ArgumentParser(description="Pico2-style local face attendance")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_register = sub.add_parser("register", help="register student templates")
    p_register.add_argument("--name", required=True)
    p_register.add_argument("--class-name", default="")
    p_register.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p_register.set_defaults(func=register)

    p_run = sub.add_parser("run", help="run local attendance")
    p_run.add_argument("--mode", choices=["len", "xuong"], default="len")
    p_run.add_argument("--threshold", type=float, default=None)
    p_run.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p_run.add_argument("--class-name", default=None, help="only recognize students from this class")
    p_run.add_argument("--detect-every", type=int, default=3)
    p_run.add_argument("--recognize-every", type=int, default=2)
    p_run.set_defaults(func=run_attendance)

    p_report = sub.add_parser("report", help="show local anomaly report")
    p_report.set_defaults(func=report)

    p_self = sub.add_parser("self-test", help="run feature/database self test")
    p_self.set_defaults(func=self_test)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
