import cv2
import numpy as np
import os
import sys
import serial
import time
import warnings
from typing import Optional

# Cấu hình output tiếng Việt
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# ============================================================
# KHỞI TẠO ỨNG DỤNG API
# ============================================================
app_api = FastAPI(
    title="API Nhận Diện Khuôn Mặt & Khóa Cửa",
    description="""
## Hệ Thống Nhận Diện Khuôn Mặt & Điều Khiển Cửa ESP32

### Các chức năng chính:
- **Đăng ký khuôn mặt**: Thêm người dùng mới vào cơ sở dữ liệu
- **Nhận diện khuôn mặt**: Xác định danh tính và tự động mở cửa
- **Quản lý người dùng**: Xem danh sách, xóa người dùng
- **Điều khiển khóa cửa**: Mở cửa thủ công qua ESP32
    """,
    version="1.0.0"
)

# ============================================================
# CẤU HÌNH CORS - Cho phép gọi API từ mọi nguồn
# (Web Frontend, AI Hub, Mobile App, v.v.)
# ============================================================
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Cho phép tất cả các nguồn
    allow_credentials=True,
    allow_methods=["*"],        # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],        # Cho phép tất cả các header
)

# ============================================================
# ĐƯỜNG DẪN CƠ SỞ DỮ LIỆU
# ============================================================
THU_MUC_GOC = os.path.dirname(os.path.abspath(__file__))
DUONG_DAN_DATABASE = os.path.join(THU_MUC_GOC, "database")
os.makedirs(DUONG_DAN_DATABASE, exist_ok=True)

# ============================================================
# KHỞI TẠO MÔ HÌNH AI (InsightFace)
# ============================================================
print("=" * 50)
print("  ĐANG KHỞI TẠO MÔ HÌNH AI INSIGHTFACE...")
print("=" * 50)
mo_hinh_ai = FaceAnalysis(providers=["CPUExecutionProvider"])
mo_hinh_ai.prepare(ctx_id=-1, det_size=(256, 256))
print("✅ Khởi tạo mô hình AI thành công!")

# ============================================================
# CẤU HÌNH KẾT NỐI ESP32 QUA SERIAL
# ============================================================
CONG_SERIAL = 'COM6'
TOC_DO_BAUD = 9600
ket_noi_serial = None

try:
    ket_noi_serial = serial.Serial(CONG_SERIAL, TOC_DO_BAUD, timeout=1)
    time.sleep(2)
    print(f"✅ Đã kết nối ESP32 qua cổng {CONG_SERIAL}")
except Exception as loi:
    print(f"⚠️  Chưa kết nối ESP32 trên {CONG_SERIAL}: {loi}")

# Thời gian hồi phục giữa 2 lần mở cửa (giây)
thoi_gian_mo_cuoi = 0
THOI_GIAN_HOI_PHUC = 3.0


# ============================================================
# CÁC HÀM TIỆN ÍCH
# ============================================================

def tai_co_so_du_lieu():
    """Đọc toàn bộ dữ liệu khuôn mặt từ thư mục database"""
    co_so_du_lieu = {}
    if os.path.exists(DUONG_DAN_DATABASE):
        for ten_file in os.listdir(DUONG_DAN_DATABASE):
            if ten_file.endswith(".npy"):
                ten_nguoi, _ = os.path.splitext(ten_file)
                vector = np.load(os.path.join(DUONG_DAN_DATABASE, ten_file))
                co_so_du_lieu[ten_nguoi] = vector
    return co_so_du_lieu


def tinh_do_tuong_dong(vec_a, vec_b):
    """Tính độ tương đồng cosine giữa 2 vector đặc trưng khuôn mặt (0.0 - 1.0)"""
    chuan_a = np.linalg.norm(vec_a)
    chuan_b = np.linalg.norm(vec_b)
    if chuan_a == 0 or chuan_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (chuan_a * chuan_b))


def gui_lenh_mo_cua():
    """Gửi lệnh OPEN tới ESP32 để mở khóa cửa"""
    global thoi_gian_mo_cuoi, ket_noi_serial
    thoi_gian_hien_tai = time.time()
    if ket_noi_serial is not None:
        if thoi_gian_hien_tai - thoi_gian_mo_cuoi > THOI_GIAN_HOI_PHUC:
            try:
                ket_noi_serial.write(b"OPEN\n")
                thoi_gian_mo_cuoi = thoi_gian_hien_tai
                print("🔓 Đã gửi lệnh MỞ CỬA tới ESP32!")
                return True
            except Exception as loi:
                print(f"❌ Lỗi gửi lệnh Serial: {loi}")
                return False
        else:
            thoi_gian_con_lai = THOI_GIAN_HOI_PHUC - (thoi_gian_hien_tai - thoi_gian_mo_cuoi)
            print(f"⏳ Đang hồi phục, thử lại sau {thoi_gian_con_lai:.1f} giây")
            return False
    return False


# ============================================================
# CÁC ENDPOINT API
# ============================================================

@app_api.get(
    "/",
    summary="Kiểm tra trạng thái server",
    tags=["Hệ thống"]
)
def kiem_tra_trang_thai():
    """Trả về thông tin trạng thái hiện tại của hệ thống"""
    co_so_du_lieu = tai_co_so_du_lieu()
    return {
        "trang_thai": "hoat_dong",
        "he_thong": "API Nhận Diện Khuôn Mặt & Khóa Cửa ESP32",
        "esp32_ket_noi": ket_noi_serial is not None and ket_noi_serial.is_open,
        "cong_serial": CONG_SERIAL,
        "so_nguoi_da_dang_ky": len(co_so_du_lieu)
    }


@app_api.get(
    "/nguoi-dung",
    summary="Xem danh sách người dùng",
    tags=["Quản lý người dùng"]
)
def xem_danh_sach_nguoi_dung():
    """Trả về danh sách tất cả người đã đăng ký khuôn mặt"""
    co_so_du_lieu = tai_co_so_du_lieu()
    return {
        "thanh_cong": True,
        "so_luong": len(co_so_du_lieu),
        "danh_sach": list(co_so_du_lieu.keys())
    }


@app_api.delete(
    "/nguoi-dung/{ten}",
    summary="Xóa người dùng",
    tags=["Quản lý người dùng"]
)
def xoa_nguoi_dung(ten: str):
    """Xóa khuôn mặt của một người dùng khỏi cơ sở dữ liệu"""
    duong_dan_file = os.path.join(DUONG_DAN_DATABASE, f"{ten}.npy")
    if os.path.exists(duong_dan_file):
        os.remove(duong_dan_file)
        return {
            "thanh_cong": True,
            "thong_bao": f"Đã xóa khuôn mặt của '{ten}' khỏi cơ sở dữ liệu"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy người dùng có tên '{ten}' trong cơ sở dữ liệu"
        )


@app_api.post(
    "/dang-ky",
    summary="Đăng ký khuôn mặt mới",
    tags=["Nhận diện khuôn mặt"]
)
async def dang_ky_khuon_mat(
    ten: str = Form(..., description="Họ và tên người cần đăng ký"),
    anh: UploadFile = File(..., description="File ảnh chứa khuôn mặt (.jpg hoặc .png)")
):
    """
    Đăng ký khuôn mặt mới vào cơ sở dữ liệu.

    - **ten**: Tên người dùng (ví dụ: Nguyễn Văn A)
    - **anh**: File ảnh chứa khuôn mặt rõ ràng, nhìn thẳng
    """
    ten = ten.strip()
    if not ten:
        raise HTTPException(status_code=400, detail="Tên người dùng không được để trống")

    # Đọc và giải mã ảnh
    du_lieu_anh = await anh.read()
    mang_numpy = np.frombuffer(du_lieu_anh, np.uint8)
    khung_anh = cv2.imdecode(mang_numpy, cv2.IMREAD_COLOR)

    if khung_anh is None:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc bị hỏng")

    # Thu nhỏ ảnh để xử lý nhanh hơn
    anh_nho = cv2.resize(khung_anh, (320, 240))
    danh_sach_khuon_mat = mo_hinh_ai.get(anh_nho)

    if len(danh_sach_khuon_mat) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không tìm thấy khuôn mặt nào trong ảnh. Vui lòng chụp lại với khuôn mặt rõ ràng hơn."
        )

    # Trích xuất vector đặc trưng và lưu vào database
    vector_dac_trung = danh_sach_khuon_mat[0].embedding
    duong_dan_luu = os.path.join(DUONG_DAN_DATABASE, f"{ten}.npy")
    np.save(duong_dan_luu, vector_dac_trung)

    print(f"✅ Đã đăng ký khuôn mặt cho: {ten}")
    return {
        "thanh_cong": True,
        "ten": ten,
        "thong_bao": f"Đã đăng ký thành công khuôn mặt cho '{ten}'",
        "duong_dan_luu": duong_dan_luu
    }


@app_api.post(
    "/nhan-dien",
    summary="Nhận diện khuôn mặt",
    tags=["Nhận diện khuôn mặt"]
)
async def nhan_dien_khuon_mat(
    tu_dong_mo_cua: bool = Form(True, description="Tự động gửi lệnh mở cửa ESP32 nếu nhận diện thành công"),
    anh: UploadFile = File(..., description="File ảnh cần nhận diện (.jpg hoặc .png)")
):
    """
    Nhận diện khuôn mặt trong ảnh và so khớp với cơ sở dữ liệu.

    - **tu_dong_mo_cua**: Nếu True, tự động gửi lệnh mở cửa khi nhận ra người
    - **anh**: File ảnh cần nhận diện
    """
    # Đọc và giải mã ảnh
    du_lieu_anh = await anh.read()
    mang_numpy = np.frombuffer(du_lieu_anh, np.uint8)
    khung_anh = cv2.imdecode(mang_numpy, cv2.IMREAD_COLOR)

    if khung_anh is None:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc bị hỏng")

    # Kiểm tra cơ sở dữ liệu
    co_so_du_lieu = tai_co_so_du_lieu()
    if len(co_so_du_lieu) == 0:
        return {
            "thanh_cong": True,
            "tim_thay": False,
            "thong_bao": "Cơ sở dữ liệu trống, chưa có ai được đăng ký",
            "ten_nhan_dien": "khong_xac_dinh",
            "do_chinh_xac": 0.0,
            "da_mo_cua": False
        }

    # Phát hiện khuôn mặt trong ảnh
    anh_nho = cv2.resize(khung_anh, (320, 240))
    danh_sach_khuon_mat = mo_hinh_ai.get(anh_nho)

    if len(danh_sach_khuon_mat) == 0:
        return {
            "thanh_cong": True,
            "tim_thay": False,
            "thong_bao": "Không phát hiện khuôn mặt nào trong ảnh",
            "ten_nhan_dien": "khong_xac_dinh",
            "do_chinh_xac": 0.0,
            "da_mo_cua": False
        }

    # So khớp với cơ sở dữ liệu
    vector_hien_tai = danh_sach_khuon_mat[0].embedding
    ten_khop_nhat = "khong_xac_dinh"
    do_chinh_xac_cao_nhat = 0.0

    for ten, vector in co_so_du_lieu.items():
        do_chinh_xac = tinh_do_tuong_dong(vector_hien_tai, vector)
        if do_chinh_xac > do_chinh_xac_cao_nhat:
            do_chinh_xac_cao_nhat = do_chinh_xac
            ten_khop_nhat = ten

    # Ngưỡng nhận diện (>= 0.5 là nhận ra)
    NGUONG_NHAN_DIEN = 0.5
    da_nhan_ra = do_chinh_xac_cao_nhat >= NGUONG_NHAN_DIEN
    ten_ket_qua = ten_khop_nhat if da_nhan_ra else "khong_xac_dinh"
    da_mo_cua = False

    # Mở cửa nếu nhận ra người
    if da_nhan_ra and tu_dong_mo_cua:
        da_mo_cua = gui_lenh_mo_cua()
        if da_nhan_ra:
            print(f"👤 Nhận diện thành công: {ten_ket_qua} (độ chính xác: {do_chinh_xac_cao_nhat:.2%})")

    return {
        "thanh_cong": True,
        "tim_thay": True,
        "da_nhan_ra": da_nhan_ra,
        "ten": ten_ket_qua,
        "do_chinh_xac": round(do_chinh_xac_cao_nhat, 4),
        "nguong": NGUONG_NHAN_DIEN,
        "da_mo_cua": da_mo_cua
    }


@app_api.post(
    "/mo-cua",
    summary="Mở cửa thủ công",
    tags=["Điều khiển khóa cửa"]
)
def mo_cua_thu_cong():
    """Gửi lệnh mở cửa trực tiếp tới ESP32 mà không cần nhận diện khuôn mặt"""
    ket_qua = gui_lenh_mo_cua()
    if ket_qua:
        return {
            "thanh_cong": True,
            "thong_bao": "Đã gửi lệnh MỞ CỬA tới ESP32 thành công"
        }
    else:
        return {
            "thanh_cong": False,
            "thong_bao": "Không thể mở cửa (Chưa kết nối ESP32 hoặc đang trong thời gian hồi phục)"
        }


# ============================================================
# ĐIỂM KHỞI CHẠY
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  KHỞI CHẠY API SERVER")
    print("=" * 50)
    print("  Địa chỉ truy cập: http://localhost:8000")
    print("  Swagger UI:       http://localhost:8000/docs")
    print("  ReDoc:            http://localhost:8000/redoc")
    print("=" * 50 + "\n")
    uvicorn.run(app_api, host="0.0.0.0", port=8000)
