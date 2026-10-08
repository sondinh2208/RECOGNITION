"""
face_engine.py - Core AI Face Engine optimized for Raspberry Pi 4B & Edge Devices
Sử dụng:
  - Face Detection: OpenCV YuNet (ONNX, ~230KB, siêu nhẹ, tối ưu ARM NEON)
  - Face Recognition: OpenCV SFace (ONNX, ~38MB, trích xuất vector 128-D)
  - Vectorized Matching: Ma trận NumPy BLAS (< 0.1ms cho 10.000 khuôn mặt)
"""

import os
import sys
import time
import urllib.request
import numpy as np
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DEFAULT_COSINE_THRESHOLD = 0.45


def imread_unicode(file_path):
    """Đọc ảnh an toàn hỗ trợ mọi đường dẫn Unicode (Windows/Linux)"""
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(file_path)


def imwrite_unicode(file_path, img):
    """Ghi ảnh an toàn hỗ trợ mọi đường dẫn Unicode (Windows/Linux)"""
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".jpg"
        ret, buf = cv2.imencode(ext, img)
        if ret:
            with open(file_path, "wb") as f:
                f.write(buf)
            return True
        return False
    except Exception:
        return cv2.imwrite(file_path, img)


def get_model_path_for_cv2(full_path, fallback_filename):
    """
    OpenCV DNN trên Windows không đọc được đường dẫn chứa ký tự Unicode có dấu (như 'Máy tính').
    Hàm này đảm bảo trả về đường dẫn tương đối an toàn hoặc chdir khi cần.
    """
    try:
        rel = os.path.relpath(full_path, os.getcwd())
        if os.path.exists(rel):
            return rel
    except Exception:
        pass
    
    # Nếu đường dẫn có ký tự non-ascii trên Windows, dùng đường dẫn tương đối từ BASE_DIR
    rel_from_base = os.path.join("models", fallback_filename)
    if os.path.exists(os.path.join(BASE_DIR, rel_from_base)):
        return os.path.join("models", fallback_filename)
        
    return full_path


def ensure_models():
    """Tự động kiểm tra và tải model ONNX nếu chưa có"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(YUNET_PATH) or os.path.getsize(YUNET_PATH) < 100000:
        print(f"[FaceEngine] Đang tải mô hình YuNet từ {YUNET_URL}...")
        try:
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            print("[FaceEngine] Tải YuNet thành công!")
        except Exception as e:
            raise RuntimeError(f"Không thể tải YuNet: {e}")

    if not os.path.exists(SFACE_PATH) or os.path.getsize(SFACE_PATH) < 30000000:
        print(f"[FaceEngine] Đang tải mô hình SFace từ {SFACE_URL}...")
        try:
            urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)
            print("[FaceEngine] Tải SFace thành công!")
        except Exception as e:
            raise RuntimeError(f"Không thể tải SFace: {e}")


class FaceEngine:
    def __init__(self, score_threshold=0.6, nms_threshold=0.3, cosine_threshold=DEFAULT_COSINE_THRESHOLD):
        ensure_models()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.cosine_threshold = cosine_threshold

        # Đảm bảo đường dẫn an toàn cho OpenCV C++
        # Nếu đang ở thư mục khác, chuyển tạm về BASE_DIR để load relative path nếu cần
        orig_cwd = os.getcwd()
        try:
            os.chdir(BASE_DIR)
            yunet_file = get_model_path_for_cv2(YUNET_PATH, "face_detection_yunet_2023mar.onnx")
            sface_file = get_model_path_for_cv2(SFACE_PATH, "face_recognition_sface_2021dec.onnx")

            # Khởi tạo YuNet Face Detector
            self.detector = cv2.FaceDetectorYN.create(
                model=yunet_file,
                config="",
                input_size=(320, 240),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000
            )

            # Khởi tạo SFace Face Recognizer
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=sface_file,
                config=""
            )
        finally:
            os.chdir(orig_cwd)

        # Database bộ nhớ
        self.db_names = []
        self.db_matrix = None  # shape: (N, 128), normalized
        self.current_input_size = (320, 240)

        print("[FaceEngine] Đã khởi tạo YuNet & SFace thành công (Tối ưu ARM NEON cho Raspberry Pi 4)!")

    def set_input_size(self, width, height):
        if self.current_input_size != (width, height):
            self.detector.setInputSize((width, height))
            self.current_input_size = (width, height)

    def detect(self, frame, input_size=None):
        """
        Phát hiện khuôn mặt trên ảnh.
        frame: BGR numpy image
        Trả về danh sách các khuôn mặt:
        [
            {
                'bbox': [x, y, w, h],
                'landmarks': [(x,y)*5],
                'score': float,
                'raw': numpy array (15 phần tử định dạng YuNet)
            }, ...
        ]
        """
        h, w = frame.shape[:2]
        if input_size is None:
            self.set_input_size(w, h)
        else:
            self.set_input_size(input_size[0], input_size[1])

        faces = self.detector.detect(frame)
        results = []
        if faces[1] is None or len(faces[1]) == 0:
            return results

        for face in faces[1]:
            score = float(face[-1])
            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
            # Clamp bbox về trong ảnh
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(w - bbox[0], bbox[2])
            bbox[3] = min(h - bbox[1], bbox[3])

            landmarks = []
            for i in range(5):
                landmarks.append((float(face[4 + i * 2]), float(face[5 + i * 2])))

            results.append({
                'bbox': bbox,
                'landmarks': landmarks,
                'score': score,
                'raw': face
            })

        return results

    def extract_feature(self, frame, raw_face):
        """
        Căn chỉnh khuôn mặt (alignCrop) và trích xuất vector đặc trưng 128 chiều.
        Tự động L2-normalize để so khớp cosine siêu tốc bằng tích vô hướng dot product.
        """
        aligned_face = self.recognizer.alignCrop(frame, raw_face)
        feature = self.recognizer.feature(aligned_face) # shape: (1, 128)
        feature = feature.flatten().astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        return feature

    def load_database(self, database_dir):
        """
        Quét và tải các vector SFace (.npy 128 chiều) từ thư mục database.
        Gom vào một ma trận (N, 128) để tìm kiếm cực nhanh.
        """
        self.db_names = []
        vectors = []

        if not os.path.exists(database_dir):
            os.makedirs(database_dir, exist_ok=True)
            self.db_matrix = None
            return 0

        legacy_512_count = 0
        for file in os.listdir(database_dir):
            if file.endswith(".npy"):
                file_path = os.path.join(database_dir, file)
                try:
                    vec = np.load(file_path).flatten().astype(np.float32)
                    if vec.shape[0] == 128:
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec /= norm
                        name = os.path.splitext(file)[0]
                        self.db_names.append(name)
                        vectors.append(vec)
                    elif vec.shape[0] == 512:
                        legacy_512_count += 1
                except Exception as e:
                    print(f"[FaceEngine] Lỗi đọc file {file}: {e}")

        if vectors:
            self.db_matrix = np.array(vectors, dtype=np.float32) # shape: (N, 128)
            print(f"[FaceEngine] Đã tải {len(self.db_names)} khuôn mặt vào bộ nhớ (Ma trận NumPy 128-D)")
        else:
            self.db_matrix = None
            print("[FaceEngine] Chưa có khuôn mặt chuẩn SFace 128-D nào trong database!")

        if legacy_512_count > 0:
            print(f"[FaceEngine] Lưu ý: Tìm thấy {legacy_512_count} file .npy 512-D (của InsightFace cũ). Hãy dùng script convert_database.py hoặc đăng ký mới qua face_embedding.py.")

        return len(self.db_names)

    def match(self, query_feature, threshold=None):
        """
        So khớp query_feature (128-D đã normalize) với database.
        Sử dụng phép nhân ma trận NumPy BLAS (db_matrix @ query_feature).
        Trả về: (best_name, confidence)
        """
        if self.db_matrix is None or len(self.db_names) == 0:
            return "unknown", 0.0

        if threshold is None:
            threshold = self.cosine_threshold

        # Tích vô hướng nhanh: similarities có shape (N,)
        similarities = np.dot(self.db_matrix, query_feature)
        best_idx = int(np.argmax(similarities))
        best_conf = float(similarities[best_idx])

        if best_conf >= threshold:
            return self.db_names[best_idx], best_conf
        else:
            return "unknown", best_conf
