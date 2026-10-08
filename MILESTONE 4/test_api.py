"""
Script kiểm tra tất cả các API endpoint của hệ thống Face ID.
Chạy: python test_api.py
(Yêu cầu API server đang chạy tại http://localhost:8000)
"""
import urllib.request
import urllib.parse
import json
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "http://localhost:8000"

def print_result(title, response_data, status_code=None):
    print(f"\n{'='*50}")
    print(f"  {title}")
    if status_code:
        print(f"  HTTP Status: {status_code}")
    print('='*50)
    print(json.dumps(response_data, ensure_ascii=False, indent=2))

def test_get(path):
    url = BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data, response.status
    except Exception as e:
        return {"error": str(e)}, None

def test_delete(path):
    url = BASE_URL + path
    req = urllib.request.Request(url, method='DELETE')
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data, response.status
    except urllib.error.HTTPError as e:
        data = json.loads(e.read().decode('utf-8'))
        return data, e.code
    except Exception as e:
        return {"error": str(e)}, None


print("\n" + "="*50)
print("  KIEM TRA API HE THONG FACE ID")
print("="*50)
print(f"  Server: {BASE_URL}")
print("="*50)

# 1. Test GET /
print("\n[1] Kiem tra server status...")
data, status = test_get("/")
print_result("GET / - Server Status", data, status)

# 2. Test GET /users
print("\n[2] Lay danh sach nguoi da dang ky...")
data, status = test_get("/users")
print_result("GET /users - Danh sach nguoi dung", data, status)

# 3. Test GET /docs (Swagger UI)
print("\n[3] Kiem tra Swagger Docs...")
try:
    with urllib.request.urlopen(BASE_URL + "/docs", timeout=5) as response:
        print(f"\nSwagger UI: OK (HTTP {response.status})")
        print(f"  Truy cap: {BASE_URL}/docs")
except Exception as e:
    print(f"\nSwagger UI: Loi - {e}")

print("\n" + "="*50)
print("  HUONG DAN TEST THEM")
print("="*50)
print("""
De test dang ky khuon mat (dung Swagger UI):
  1. Mo trinh duyet tai: http://localhost:8000/docs
  2. Chon POST /register
  3. Click 'Try it out'
  4. Nhap name va chon file anh
  5. Click 'Execute'

Hoac dung curl:
  curl -X POST "http://localhost:8000/register" ^
       -F "name=TenNguoi" ^
       -F "image=@duong/dan/anh.jpg"
""")

print("\nKiem tra co ban hoan tat!")
print(f"Swagger UI: {BASE_URL}/docs")
print(f"ReDoc:      {BASE_URL}/redoc")
