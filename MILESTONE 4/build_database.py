import cv2
from insightface.app import FaceAnalysis

# Khởi tạo InsightFace
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(1024, 1024))

# Đọc ảnh
img = cv2.imread("database/0065.png")

print(img.shape)
print(img.dtype)

# Detector
bboxes, kpss = app.det_model.detect(img, max_num=1)

print("Boxes:", bboxes)

# FaceAnalysis
faces = app.get(img)
print("So mat:", len(faces))