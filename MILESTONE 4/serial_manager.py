"""
serial_manager.py - ESP32 Serial Communication Manager
Hỗ trợ tự động nhận diện cổng nối tiếp trên cả Linux (Raspberry Pi 4) và Windows.
Quản lý gửi lệnh OPEN không nghẽn luồng (Non-blocking Cooldown).
"""

import sys
import glob
import time
import threading
import serial

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def find_serial_ports():
    """Tự động tìm kiếm các cổng COM/Serial khả dụng"""
    ports = []
    if sys.platform.startswith('win'):
        # Quét cổng COM trên Windows
        for i in range(1, 20):
            port = f"COM{i}"
            try:
                s = serial.Serial(port)
                s.close()
                ports.append(port)
            except (OSError, serial.SerialException):
                pass
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # Quét /dev/ttyUSB* và /dev/ttyACM* trên Linux / Raspberry Pi
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.usb*')
    return ports


class SerialManager:
    def __init__(self, port=None, baudrate=9600, cooldown=4.0):
        self.baudrate = baudrate
        self.cooldown = cooldown
        self.last_open_time = 0
        self.ser = None
        self.lock = threading.Lock()

        # Tự động chọn cổng nếu không chỉ định
        target_port = port
        if target_port is None:
            available = find_serial_ports()
            if available:
                target_port = available[0]
                print(f"[Serial] Tìm thấy các cổng: {available}, tự động chọn: {target_port}")
            else:
                # Mặc định theo hệ điều hành
                target_port = "/dev/ttyUSB0" if sys.platform.startswith('linux') else "COM6"

        self.port = target_port
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(1.5)  # Chờ ESP32 ổn định sau khi mở kết nối
            print(f"[Serial] ✅ Đã kết nối ESP32 qua cổng {self.port} (Baud: {self.baudrate})")
        except Exception as e:
            self.ser = None
            print(f"[Serial] ⚠️ Chưa thể kết nối ESP32 tại {self.port}: {e}")
            print("         Hệ thống sẽ chạy ở chế độ mô phỏng (không điều khiển servo vật lý).")

    def send_open(self, trigger_name=""):
        """Gửi lệnh OPEN tới ESP32 nếu ngoài thời gian cooldown"""
        current_time = time.time()
        if current_time - self.last_open_time < self.cooldown:
            return False, "COOLDOWN"

        self.last_open_time = current_time
        if self.ser is not None and self.ser.is_open:
            try:
                with self.lock:
                    self.ser.write(b"OPEN\n")
                    self.ser.flush()
                print(f"[Serial] 🚪 Đã gửi lệnh OPEN tới ESP32 (Nhận diện: {trigger_name})")
                return True, "SENT"
            except Exception as e:
                print(f"[Serial] Lỗi gửi dữ liệu serial: {e}")
                return False, f"ERROR: {e}"
        else:
            print(f"[Serial] [Mô phỏng] 🚪 Lệnh OPEN được kích hoạt cho: {trigger_name}")
            return True, "SIMULATED"

    def close(self):
        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.close()
                print("[Serial] Đã đóng kết nối ESP32.")
            except Exception:
                pass
