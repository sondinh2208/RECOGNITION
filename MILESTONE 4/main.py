import subprocess
import sys
import os
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    while True:
        print("\n" + "="*50)
        print("   HỆ THỐNG MỞ CỬA NHẬN DIỆN KHUÔN MẶT")
        print("="*50)
        print("1. Đăng ký khuôn mặt mới")
        print("2. Nhận diện & mở cửa (Real-time Tracker)")
        print("3. Xem danh sách người đã đăng ký")
        print("4. Xóa khuôn mặt trong database")
        print("5. Khởi chạy REST API Server (Cho AI Hub / Web)")
        print("6. Cập nhật / Chuyển đổi toàn bộ ảnh sang SFace 128-D")
        print("7. Thoát")
        print("="*50)
        
        choice = input("Chọn chức năng (1-7): ").strip()
        
        if choice == '1':
            print("\n--- ĐĂNG KÝ KHUÔN MẶT ---")
            script_path = os.path.join(BASE_DIR, "face_embedding.py")
            subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
        elif choice == '2':
            print("\n--- NHẬN DIỆN & MỞ CỬA ---")
            script_path = os.path.join(BASE_DIR, "recognition.py")
            subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
        elif choice == '3':
            print("\n--- DANH SÁCH NGƯỜI ĐÃ ĐĂNG KÝ ---")
            db_path = os.path.join(BASE_DIR, "database")
            if os.path.exists(db_path):
                files = [f for f in os.listdir(db_path) if f.endswith('.npy')]
                if files:
                    print(f"Tổng số người đã đăng ký: {len(files)}")
                    for idx, f in enumerate(files, 1):
                        name = os.path.splitext(f)[0]
                        print(f"  {idx}. {name}")
                else:
                    print("  Chưa có người đăng ký trong database!")
            else:
                print("  Thư mục database không tồn tại!")
            input("\nNhấn Enter để quay lại menu...")
        elif choice == '4':
            print("\n--- XÓA KHUÔN MẶT ---")
            name = input("Nhập tên cần xóa: ").strip()
            if name:
                file_path = os.path.join(BASE_DIR, "database", f"{name}.npy")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Đã xóa thành công: {name}")
                else:
                    print(f"Không tìm thấy người có tên: {name}")
            else:
                print("Tên nhập vào không hợp lệ!")
            input("\nNhấn Enter để quay lại menu...")
        elif choice == '5':
            print("\n--- KHỞI CHẠY REST API SERVER ---")
            script_path = os.path.join(BASE_DIR, "api.py")
            subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
        elif choice == '6':
            print("\n--- CHUYỂN ĐỔI DATABASE SANG SFACE 128-D ---")
            script_path = os.path.join(BASE_DIR, "convert_database.py")
            subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
            input("\nNhấn Enter để quay lại menu...")
        elif choice == '7':
            print("Thoát chương trình!")
            break
        else:
            print("Lựa chọn không hợp lệ!")
        
        time.sleep(0.05)

if __name__ == "__main__":
    main()