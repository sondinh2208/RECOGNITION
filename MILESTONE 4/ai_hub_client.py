"""
=============================================================
  CLIENT MẪU CHO TEAM AI HUB
  Hệ Thống Nhận Diện Khuôn Mặt & Khóa Cửa ESP32
=============================================================

Cài thư viện: pip install requests
Chạy:         python ai_hub_client.py
=============================================================
"""
import requests
import sys

# ==== CẤU HÌNH - ĐỔI IP NÀY THÀNH IP MÁY CỦA TO MANH HUNG ====
URL_SERVER = "http://192.168.3.101:8000"
# ==============================================================


def kiem_tra_ket_noi():
    """Kiểm tra server có đang chạy không"""
    try:
        res = requests.get(f"{URL_SERVER}/", timeout=5)
        data = res.json()
        print(f"✅ Server đang hoạt động")
        print(f"   ESP32 kết nối: {data['esp32_ket_noi']}")
        print(f"   Số người đăng ký: {data['so_nguoi_da_dang_ky']}")
        return True
    except Exception as e:
        print(f"❌ Không kết nối được server: {e}")
        print(f"   Kiểm tra lại: server có đang chạy không? IP có đúng không?")
        return False


def xem_danh_sach_nguoi_dung():
    """Lấy danh sách tất cả người đã đăng ký"""
    res = requests.get(f"{URL_SERVER}/nguoi-dung")
    data = res.json()
    print(f"\n📋 Danh sách người dùng ({data['so_luong']} người):")
    for i, nguoi in enumerate(data["danh_sach"], 1):
        print(f"   {i}. {nguoi}")
    return data["danh_sach"]


def dang_ky_khuon_mat(ten, duong_dan_anh):
    """
    Đăng ký khuôn mặt mới vào hệ thống

    Tham số:
        ten           : Tên người dùng (str)
        duong_dan_anh : Đường dẫn file ảnh .jpg hoặc .png (str)

    Trả về: dict kết quả từ server
    """
    print(f"\n📸 Đang đăng ký khuôn mặt cho: {ten}")
    try:
        with open(duong_dan_anh, "rb") as file_anh:
            res = requests.post(
                f"{URL_SERVER}/dang-ky",
                data={"ten": ten},
                files={"anh": ("anh.jpg", file_anh, "image/jpeg")}
            )
        data = res.json()
        if res.status_code == 200 and data.get("thanh_cong"):
            print(f"✅ Đăng ký thành công: {ten}")
        else:
            print(f"❌ Lỗi đăng ký: {data.get('detail', 'Không rõ lỗi')}")
        return data
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file ảnh: {duong_dan_anh}")
        return None


def nhan_dien_khuon_mat(duong_dan_anh, tu_dong_mo_cua=True):
    """
    Nhận diện khuôn mặt trong ảnh và tùy chọn mở cửa

    Tham số:
        duong_dan_anh  : Đường dẫn file ảnh .jpg hoặc .png (str)
        tu_dong_mo_cua : True = tự động mở cửa nếu nhận ra người (bool)

    Trả về: dict gồm:
        - da_nhan_ra   : True/False
        - ten          : Tên người nhận ra ("khong_xac_dinh" nếu không nhận ra)
        - do_chinh_xac : Độ chính xác từ 0.0 đến 1.0
        - da_mo_cua    : True/False
    """
    print(f"\n🔍 Đang nhận diện khuôn mặt từ: {duong_dan_anh}")
    try:
        with open(duong_dan_anh, "rb") as file_anh:
            res = requests.post(
                f"{URL_SERVER}/nhan-dien",
                data={"tu_dong_mo_cua": str(tu_dong_mo_cua).lower()},
                files={"anh": ("anh.jpg", file_anh, "image/jpeg")}
            )
        data = res.json()

        if not data.get("tim_thay"):
            print(f"⚠️  Không phát hiện khuôn mặt trong ảnh")
        elif data.get("da_nhan_ra"):
            do_cx = data['do_chinh_xac']
            print(f"✅ Nhận ra: {data['ten']} (độ chính xác: {do_cx:.1%})")
            if data.get("da_mo_cua"):
                print(f" Đã mở cửa!")
            elif tu_dong_mo_cua:
                print(f" Chưa mở được cửa (ESP32 chưa kết nối hoặc đang hồi phục)")
        else:
            do_cx = data['do_chinh_xac']
            print(f"❌ Không nhận ra (độ tương đồng: {do_cx:.1%}, ngưỡng: {data['nguong']:.1%})")

        return data
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file ảnh: {duong_dan_anh}")
        return None


def mo_cua_thu_cong():
    """Mở cửa thủ công không cần nhận diện khuôn mặt"""
    print("\n Đang gửi lệnh mở cửa...")
    res = requests.post(f"{URL_SERVER}/mo-cua")
    data = res.json()
    if data.get("thanh_cong"):
        print("✅ Đã mở cửa thành công!")
    else:
        print(f"  {data.get('thong_bao')}")
    return data


def xoa_nguoi_dung(ten):
    """Xóa người dùng khỏi cơ sở dữ liệu"""
    print(f"\n  Đang xóa người dùng: {ten}")
    res = requests.delete(f"{URL_SERVER}/nguoi-dung/{ten}")
    data = res.json()
    if data.get("thanh_cong"):
        print(f"✅ Đã xóa: {ten}")
    else:
        print(f" Lỗi: {data.get('detail')}")
    return data


# ==============================================================
# CHƯƠNG TRÌNH DEMO
# ==============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  DEMO CLIENT - HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")
    print("=" * 55)
    print(f"  Server: {URL_SERVER}")
    print("=" * 55)

    # Bước 1: Kiểm tra kết nối
    if not kiem_tra_ket_noi():
        sys.exit(1)

    # Bước 2: Xem danh sách người dùng
    xem_danh_sach_nguoi_dung()

    print("\n" + "=" * 55)
    print("  Chỉnh sửa file này để gọi các hàm theo nhu cầu:")
    print("    dang_ky_khuon_mat('Ten Nguoi', 'anh.jpg')")
    print("    nhan_dien_khuon_mat('anh.jpg')")
    print("    mo_cua_thu_cong()")
    print("=" * 55)
