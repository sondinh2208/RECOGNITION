"""
api.py - REST API Server Nhận diện khuôn mặt & Mở cửa ESP32
Tối ưu hóa siêu tốc cho Raspberry Pi 4B & Edge AI với OpenCV YuNet + SFace.
"""

import os
import sys
import time
import warnings
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from face_engine import FaceEngine, imwrite_unicode
from serial_manager import SerialManager

warnings.filterwarnings('ignore')

# ============================================================
# KHỞI TẠO ỨNG DỤNG FASTAPI
# ============================================================
app_api = FastAPI(
    title="API Nhận Diện Khuôn Mặt & Khóa Cửa (Tối ưu Pi 4)",
    description="""
## Hệ Thống Nhận Diện Khuôn Mặt & Điều Khiển Cửa ESP32
Được tối ưu hóa bằng mô hình YuNet + SFace đạt hiệu năng cao trên Raspberry Pi 4B.
    """,
    version="2.0.0"
)

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_PATH, exist_ok=True)

# Khởi tạo FaceEngine và SerialManager
print("=" * 55)
print("  ĐANG KHỞI TẠO AI ENGINE (YUNET + SFACE) CHO PI 4...")
print("=" * 55)
engine = FaceEngine(score_threshold=0.6, cosine_threshold=0.42)
engine.load_database(DATABASE_PATH)
serial_mgr = SerialManager(cooldown=4.0)
print("✅ Khởi tạo hệ thống thành công!")


# ============================================================
# CÁC ENDPOINT API
# ============================================================

@app_api.get("/", summary="Kiểm tra trạng thái server", tags=["Hệ thống"])
def kiem_tra_trang_thai():
    registered_count = len(engine.db_names)
    return {
        "trang_thai": "hoat_dong",
        "he_thong": "API Nhan Dien Khuon Mat & Khoa Cua (YuNet + SFace - Pi 4)",
        "esp32_ket_noi": serial_mgr.ser is not None and serial_mgr.ser.is_open,
        "cong_serial": serial_mgr.port,
        "so_nguoi_da_dang_ky": registered_count
    }


@app_api.get("/nguoi-dung", summary="Xem danh sách người dùng", tags=["Quản lý người dùng"])
@app_api.get("/users", summary="Xem danh sách người dùng (alias)", tags=["Quản lý người dùng"], include_in_schema=False)
def xem_danh_sach_nguoi_dung():
    engine.load_database(DATABASE_PATH)
    return {
        "thanh_cong": True,
        "so_luong": len(engine.db_names),
        "danh_sach": engine.db_names
    }


@app_api.delete("/nguoi-dung/{ten}", summary="Xóa người dùng", tags=["Quản lý người dùng"])
@app_api.delete("/users/{ten}", summary="Xóa người dùng (alias)", tags=["Quản lý người dùng"], include_in_schema=False)
def xoa_nguoi_dung(ten: str):
    ten = ten.strip()
    npy_path = os.path.join(DATABASE_PATH, f"{ten}.npy")
    jpg_path = os.path.join(DATABASE_PATH, f"{ten}.jpg")
    png_path = os.path.join(DATABASE_PATH, f"{ten}.png")

    deleted = False
    for p in [npy_path, jpg_path, png_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
                deleted = True
            except Exception:
                pass

    if deleted:
        engine.load_database(DATABASE_PATH)
        return {
            "thanh_cong": True,
            "thong_bao": f"Đã xóa khuôn mặt của '{ten}' khỏi cơ sở dữ liệu"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy người dùng có tên '{ten}' trong cơ sở dữ liệu"
        )


@app_api.post("/dang-ky", summary="Đăng ký khuôn mặt mới", tags=["Nhận diện khuôn mặt"])
@app_api.post("/register", summary="Đăng ký khuôn mặt mới (alias)", tags=["Nhận diện khuôn mặt"], include_in_schema=False)
async def dang_ky_khuon_mat(
    ten: str = Form(..., description="Họ và tên người cần đăng ký"),
    anh: UploadFile = File(..., description="File ảnh chứa khuôn mặt (.jpg hoặc .png)")
):
    ten = ten.strip()
    if not ten:
        raise HTTPException(status_code=400, detail="Tên người dùng không được để trống")

    du_lieu_anh = await anh.read()
    mang_numpy = np.frombuffer(du_lieu_anh, np.uint8)
    khung_anh = cv2.imdecode(mang_numpy, cv2.IMREAD_COLOR)

    if khung_anh is None:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc bị hỏng")

    h, w = khung_anh.shape[:2]
    faces = engine.detect(khung_anh, input_size=(w, h))

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không tìm thấy khuôn mặt nào trong ảnh. Vui lòng chụp lại với khuôn mặt rõ ràng hơn."
        )

    best_face = max(faces, key=lambda f: f['score'])
    feat = engine.extract_feature(khung_anh, best_face['raw'])

    npy_path = os.path.join(DATABASE_PATH, f"{ten}.npy")
    jpg_path = os.path.join(DATABASE_PATH, f"{ten}.jpg")

    np.save(npy_path, feat)
    imwrite_unicode(jpg_path, khung_anh)

    # Nạp lại database
    engine.load_database(DATABASE_PATH)

    print(f"✅ Đã đăng ký khuôn mặt cho: {ten}")
    return {
        "thanh_cong": True,
        "ten": ten,
        "thong_bao": f"Đã đăng ký thành công khuôn mặt cho '{ten}'",
        "duong_dan_luu": npy_path
    }


@app_api.post("/nhan-dien", summary="Nhận diện khuôn mặt", tags=["Nhận diện khuôn mặt"])
@app_api.post("/recognize", summary="Nhận diện khuôn mặt (alias)", tags=["Nhận diện khuôn mặt"], include_in_schema=False)
async def nhan_dien_khuon_mat(
    tu_dong_mo_cua: bool = Form(True, description="Tự động gửi lệnh mở cửa ESP32 nếu nhận diện thành công"),
    anh: UploadFile = File(..., description="File ảnh cần nhận diện (.jpg hoặc .png)")
):
    du_lieu_anh = await anh.read()
    mang_numpy = np.frombuffer(du_lieu_anh, np.uint8)
    khung_anh = cv2.imdecode(mang_numpy, cv2.IMREAD_COLOR)

    if khung_anh is None:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc bị hỏng")

    if len(engine.db_names) == 0:
        return {
            "thanh_cong": True,
            "tim_thay": False,
            "thong_bao": "Cơ sở dữ liệu trống, chưa có ai được đăng ký",
            "ten_nhan_dien": "khong_xac_dinh",
            "do_chinh_xac": 0.0,
            "da_mo_cua": False
        }

    h, w = khung_anh.shape[:2]
    faces = engine.detect(khung_anh, input_size=(w, h))

    if len(faces) == 0:
        return {
            "thanh_cong": True,
            "tim_thay": False,
            "thong_bao": "Không phát hiện khuôn mặt nào trong ảnh",
            "ten_nhan_dien": "khong_xac_dinh",
            "do_chinh_xac": 0.0,
            "da_mo_cua": False
        }

    best_face = max(faces, key=lambda f: f['score'])
    feat = engine.extract_feature(khung_anh, best_face['raw'])

    best_name, best_conf = engine.match(feat)
    da_nhan_ra = (best_name != "unknown")
    da_mo_cua = False

    if da_nhan_ra and tu_dong_mo_cua:
        success, _ = serial_mgr.send_open(trigger_name=best_name)
        da_mo_cua = success
        print(f"👤 Nhận diện thành công: {best_name} (độ khớp: {best_conf:.2%})")

    return {
        "thanh_cong": True,
        "tim_thay": True,
        "da_nhan_ra": da_nhan_ra,
        "ten": best_name if da_nhan_ra else "khong_xac_dinh",
        "do_chinh_xac": round(float(best_conf), 4),
        "nguong": engine.cosine_threshold,
        "da_mo_cua": da_mo_cua
    }


@app_api.post("/mo-cua", summary="Mở cửa thủ công", tags=["Điều khiển khóa cửa"])
@app_api.post("/open-door", summary="Mở cửa thủ công (alias)", tags=["Điều khiển khóa cửa"], include_in_schema=False)
def mo_cua_thu_cong():
    success, status = serial_mgr.send_open(trigger_name="Manual API")
    if success:
        return {
            "thanh_cong": True,
            "thong_bao": f"Đã gửi lệnh MỞ CỬA tới ESP32 thành công ({status})"
        }
    else:
        return {
            "thanh_cong": False,
            "thong_bao": f"Không thể mở cửa ({status})"
        }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  KHỞI CHẠY API SERVER (TỐI ƯU PI 4B)")
    print("=" * 55)
    print("  Địa chỉ:    http://0.0.0.0:8000")
    print("  Swagger UI: http://localhost:8000/docs")
    print("=" * 55 + "\n")
    uvicorn.run(app_api, host="0.0.0.0", port=8000)
