import tempfile
from pathlib import Path

import cv2
import numpy as np

import pico2_local as p


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        p.DB_DIR = root / "db"
        p.PHOTO_DIR = root / "photos"
        p.SQLITE_PATH = root / "attendance.sqlite"
        p.CONFIG_PATH = root / "config.json"
        p.ensure_dirs()
        conn = p.init_db()

        face_a = np.zeros((p.FACE_SIZE, p.FACE_SIZE), dtype=np.uint8)
        cv2.circle(face_a, (24, 18), 9, 190, -1)
        cv2.rectangle(face_a, (14, 30), (34, 43), 130, -1)

        face_b = np.zeros((p.FACE_SIZE, p.FACE_SIZE), dtype=np.uint8)
        cv2.circle(face_b, (17, 18), 7, 210, -1)
        cv2.circle(face_b, (31, 18), 7, 110, -1)
        cv2.rectangle(face_b, (10, 33), (38, 42), 170, -1)

        feat_a = p.extract_feature(face_a)
        feat_b = p.extract_feature(face_b)
        p.save_student("Hoc Sinh A", "7A1", [feat_a])
        p.save_student("Hoc Sinh B", "7A2", [feat_b])
        students = p.load_students()

        name_a, class_a, score_a = p.recognize(feat_a, students)
        name_b, class_b, score_b = p.recognize(feat_b, students)

        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        for _ in range(12):
            p.save_rolling_photo("Hoc Sinh A", frame)
        photo_count = len(list((p.PHOTO_DIR / "Hoc Sinh A").glob("*.jpg")))

        photo_path = str(next((p.PHOTO_DIR / "Hoc Sinh A").glob("*.jpg")))
        p.record_attendance(conn, "Hoc Sinh A", "7A1", "len", 0.91, photo_path)
        anomaly_1 = p.detect_anomaly(conn, "Hoc Sinh A", "len")
        p.record_attendance(conn, "Hoc Sinh A", "7A1", "xuong", 0.90, photo_path)
        anomaly_2 = p.detect_anomaly(conn, "Hoc Sinh A", "xuong")
        duplicate_off_allowed = p.should_record_attendance(conn, "Hoc Sinh A", "7A1", "xuong")
        p.record_attendance(conn, "Hoc Sinh B", "7A2", "xuong", 0.86, photo_path)
        anomaly_3 = p.detect_anomaly(conn, "Hoc Sinh B", "xuong")
        if anomaly_3[0]:
            p.save_anomaly(conn, "Hoc Sinh B", anomaly_3[0], anomaly_3[1])

        rows = conn.execute("SELECT student_name, mode, confidence FROM attendance ORDER BY id").fetchall()
        anomaly_rows = conn.execute("SELECT student_name, type FROM anomalies ORDER BY id").fetchall()
        conn.close()

        passed = (
            name_a == "Hoc Sinh A"
            and class_a == "7A1"
            and name_b == "Hoc Sinh B"
            and class_b == "7A2"
            and score_a > 0.99
            and score_b > 0.99
            and photo_count == 10
            and anomaly_1 == (None, None)
            and anomaly_2 == (None, None)
            and duplicate_off_allowed is False
            and anomaly_3[0] == "off_without_board"
            and len(rows) == 3
            and len(anomaly_rows) == 1
        )

        print(f"RECOGNIZE_A={name_a},{class_a},{score_a:.4f}")
        print(f"RECOGNIZE_B={name_b},{class_b},{score_b:.4f}")
        print(f"PHOTO_COUNT={photo_count}")
        print(f"ANOMALY_AFTER_LEN={anomaly_1}")
        print(f"ANOMALY_AFTER_MATCHED_OFF={anomaly_2}")
        print(f"DUPLICATE_OFF_ALLOWED={duplicate_off_allowed}")
        print(f"ANOMALY_OFF_WITHOUT_BOARD={anomaly_3}")
        print(f"ATTENDANCE_ROWS={rows}")
        print(f"ANOMALY_ROWS={anomaly_rows}")
        print(f"PASS={passed}")
        raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
