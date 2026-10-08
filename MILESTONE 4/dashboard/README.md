# Dashboard - Hệ Thống Nhận Diện Khuôn Mặt

## Cách sử dụng

### 1. Khởi động API Server
Trước tiên, cần khởi động API server từ thư mục gốc dự án:

```bash
python api.py
```

Server sẽ chạy tại: http://localhost:8000

### 2. Mở Dashboard
Mở file `index.html` bằng trình duyệt web (Chrome, Firefox, Edge...)

**Lưu ý:** Do API có CORS được bật, bạn có thể mở file trực tiếp hoặc dùng server tĩnh đơn giản:

```bash
# Cài đặt nếu chưa có
npm install -g serve

# Chạy server
cd dashboard
serve
```

## Các chức năng chính

### 📊 Thống kê hệ thống
- Số người đăng ký
- Trạng thái kết nối ESP32
- Trạng thái cửa
- Lần truy cập cuối

### 👤 Đăng ký khuôn mặt (Ảnh)
1. Nhập họ và tên người dùng
2. Chọn ảnh có chứa khuôn mặt (JPG/PNG)
3. Nhấn nút "Đăng ký"

### 📹 Đăng ký khuôn mặt (Video)
1. Nhập họ và tên người dùng
2. Click "Bật Camera" và cho phép truy cập camera
3. Nhìn thẳng vào camera
4. Click "Chụp ảnh" để đăng ký

### 🔍 Nhận diện khuôn mặt (Ảnh)
1. Chọn ảnh cần nhận diện
2. Bật/tắt tùy chọn "Tự động mở cửa"
3. Nhấn nút "Nhận diện"

### 📹 Nhận diện khuôn mặt (Video)
1. Click "Bật Camera" và cho phép truy cập camera
2. Nhìn thẳng vào camera
3. Click "Nhận diện ngay" để xác thực

### 👥 Quản lý người dùng
- Xem danh sách người đã đăng ký
- Xóa người dùng
- Mở cửa thủ công

## API Endpoints

| Endpoint | Phương thức | Mô tả |
|----------|-------------|------|
| `/` | GET | Kiểm tra trạng thái server |
| `/nguoi-dung` | GET | Lấy danh sách người dùng |
| `/nguoi-dung/{ten}` | DELETE | Xóa người dùng |
| `/dang-ky` | POST | Đăng ký khuôn mặt mới |
| `/nhan-dien` | POST | Nhận diện khuôn mặt |
| `/mo-cua` | POST | Mở cửa thủ công |

## Cấu trúc thư mục

```
dashboard/
├── index.html    # Giao diện dashboard
├── style.css     # Stylesheet
├── script.js     # JavaScript xử lý
└── README.md     # Hướng dẫn này