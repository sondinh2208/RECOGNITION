 Hệ thống mở cửa nhận diện khuôn mặt

Cấu trúc dự án
Face_ID/
├── main.py           # Menu chính điều khiển
├── face_embedding.py   # Đăng ký khuôn mặt
├── recognition.py    # Nhận diện & mở cửa
├── door_lock.ino      # Code ESP32
├── database/          # Cơ sở dữ liệu khuôn mặt
│   └── Hung.npy       # Dữ liệu khuôn mặt mẫu
└── README.md         # Hướng dẫn

Sử dụng
 Chạy chương trình
bash
cd Face_ID
python main.py


Menu chức năng:
1. Đăng ký khuôn mặt mới- Chạy camera, nhấn 'S' để lưu khuôn mặt
2. Nhận diện & mở cửa - Nhận diện khuôn mặt, tự động mở cửa qua ESP32
3. Xem danh sách người đã đăng ký** - Liệt kê người trong database
4. Xóa khuôn mặt - Xóa người khỏi database
5. Thoát**

 Nạp code ESP32

1. Mở Arduino IDE
2. Cài đặt board ESP32
3. Mở file `door_lock.ino`
4. Chọn board: ESP32 Dev Module
5. Chọn cổng COM đúng
6. Nạp code

#Phần cứng

- ESP32
- Servo motor (SG90)
- Dây nối

 Kết nối:
- Servo GND -> ESP32 GND
- Servo VCC -> ESP32 5V
- Servo Signal -> ESP32 GPIO 13

Lưu ý

- Ngưỡng nhận diện: 0.5 (có thể điều chỉnh trong recognition.py)
- Thời gian mở cửa tự động khóa lại: 5 giây
- Đảm bảo ánh sáng tốt khi đăng ký và nhận diện