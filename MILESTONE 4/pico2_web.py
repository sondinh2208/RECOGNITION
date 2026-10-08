import json
import shutil
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import pico2_local as core


HOST = "127.0.0.1"
PORT = 8088


class CameraWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.scan_enabled = False
        self.thread = None
        self.mode = "len"
        self.active_class = ""
        self.last_jpeg = None
        self.last_frame = None
        self.last_result = {
            "name": "unknown",
            "class_name": "",
            "score": 0.0,
            "message": "Camera is stopped",
        }
        self.last_seen = {}
        self.students = {}
        self.config = core.load_config()
        self.threshold = float(self.config.get("threshold", core.DEFAULT_THRESHOLD))

    def start(self):
        if self.running:
            return
        self.students = core.load_students()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.scan_enabled = False
        self.running = False

    def set_scan(self, enabled):
        self.scan_enabled = bool(enabled)
        if self.scan_enabled:
            self.students = core.load_students(self.active_class or None)

    def set_mode(self, mode):
        if mode in ("len", "xuong"):
            self.mode = mode

    def set_active_class(self, class_name):
        self.active_class = class_name.strip()
        self.students = core.load_students(self.active_class or None)

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "scan_enabled": self.scan_enabled,
                "mode": self.mode,
                "active_class": self.active_class,
                "last_result": dict(self.last_result),
                "student_count": len(self.students),
                "classes": core.list_classes(),
            }

    def register_current_face(self, name, class_name):
        class_name = class_name.strip()
        if not class_name:
            return False, "Hay tao/chon lop truoc"

        with self.lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
        if frame is None:
            return False, "Camera chua co frame"

        detector = core.load_detector()
        bbox, _ = core.detect_largest_face(detector, frame)
        if bbox is None:
            return False, "Khong thay mat trong khung hinh"

        face = core.preprocess_face(frame, bbox)
        if face is None:
            return False, "Khong crop duoc mat"

        feature = core.extract_feature(face)
        old = core.load_student_features(name, class_name)
        templates = [feature] if old is None else list(old)[-9:] + [feature]
        core.save_student(name, class_name, templates)
        self.students = core.load_students()
        return True, f"Da luu mau {len(templates)}/10 cho {name} - lop {class_name}"

    def _loop(self):
        detector = core.load_detector()
        conn = core.init_db()
        cam = None
        cached_bbox = None
        cached_result = ("unknown", "", 0.0)
        frame_id = 0

        try:
            cam = core.open_camera(int(self.config.get("camera_index", 0)))
            while self.running:
                ok, frame = cam.read()
                if not ok:
                    continue

                frame_id += 1
                should_detect = cached_bbox is None or frame_id % 3 == 0
                if should_detect:
                    cached_bbox, _ = core.detect_largest_face(detector, frame)

                if self.scan_enabled and cached_bbox is not None and frame_id % 2 == 0:
                    face = core.preprocess_face(frame, cached_bbox)
                    if face is not None and self.students:
                        feature = core.extract_feature(face)
                        name, class_name, score = core.recognize(feature, self.students)
                        if score < self.threshold:
                            name, class_name = "unknown", ""
                        cached_result = (name, class_name, score)
                        if name != "unknown":
                            self._record_if_needed(conn, name, class_name, score, frame)
                elif not self.scan_enabled:
                    cached_result = ("unknown", "", 0.0)

                self._draw_overlay(frame, cached_bbox, cached_result)
                ok_jpeg, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok_jpeg:
                    with self.lock:
                        self.last_frame = frame.copy()
                        self.last_jpeg = buffer.tobytes()
                        name, class_name, score = cached_result
                        self.last_result = {
                            "name": name,
                            "class_name": class_name,
                            "score": round(float(score), 4),
                            "message": "Recognizing" if self.scan_enabled else "Registration camera",
                        }
        except Exception as exc:
            with self.lock:
                self.last_result = {
                    "name": "unknown",
                    "class_name": "",
                    "score": 0.0,
                    "message": str(exc),
                }
        finally:
            if cam is not None:
                cam.release()
            conn.close()
            self.running = False
            self.scan_enabled = False

    def _record_if_needed(self, conn, name, class_name, score, frame):
        guard_key = f"{self.mode}:{class_name}:{name}"
        now = time.time()
        if now - self.last_seen.get(guard_key, 0.0) <= core.REPEAT_GUARD_SECONDS:
            return

        if not core.should_record_attendance(conn, name, class_name, self.mode):
            return

        photo_path = core.save_rolling_photo(f"{class_name}_{name}", frame)
        core.record_attendance(conn, name, class_name, self.mode, score, photo_path)
        anomaly_type, message = core.detect_anomaly(conn, name, self.mode, class_name)
        if anomaly_type:
            core.save_anomaly(conn, name, anomaly_type, message)
            core.send_anomaly_if_configured(self.config, name, anomaly_type, message)
        self.last_seen[guard_key] = now

    def _draw_overlay(self, frame, bbox, result):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            name, class_name, score = result
            color = (0, 220, 0) if name != "unknown" else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            safe_name = core.display_label(name)
            safe_class = core.display_label(class_name) if class_name else ""
            label = f"{safe_class}/{safe_name}:{score:.2f}" if safe_class else f"{safe_name}:{score:.2f}"
            cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        status = "SCAN" if self.scan_enabled else "REGISTER"
        cv2.putText(frame, f"{status} MODE:{self.mode.upper()}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)


worker = CameraWorker()


HTML = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BusCheck Local Classes</title>
  <style>
    body{margin:0;background:#eef2f6;color:#152232;font-family:Segoe UI,Arial,sans-serif}
    .shell{display:grid;grid-template-columns:260px 1fr;min-height:100vh}
    aside{background:#132433;color:white;padding:22px}
    h1,h2,p{margin:0}.brand{font-size:24px;font-weight:800}.sub{color:#a8b7c4;margin-top:5px}
    .nav{display:grid;gap:10px;margin-top:26px}.nav button{border:0;border-radius:8px;padding:12px;text-align:left;background:#20384b;color:white;cursor:pointer}
    main{padding:24px}.top{display:flex;justify-content:space-between;gap:16px;margin-bottom:18px}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
    .card,.metric{background:white;border:1px solid #dce3ea;border-radius:8px;box-shadow:0 16px 40px #1f2e3c18}
    .metric{padding:15px}.metric span{color:#687483}.metric strong{display:block;font-size:30px;margin-top:6px}
    .metric.warn{border-color:#d33b21;background:#fff4ef}.metric.warn strong{color:#d33b21}
    .grid{display:grid;grid-template-columns:minmax(360px,1.35fr) minmax(320px,.9fr);gap:16px}.card{padding:17px}
    .stream{width:100%;aspect-ratio:4/3;object-fit:cover;background:#13202c;border-radius:8px;margin:12px 0}
    .row{display:flex;gap:8px;flex-wrap:wrap}.btn{border:0;border-radius:8px;padding:10px 13px;color:white;background:#1f6fb2;cursor:pointer}
    .btn.alt{background:#177d87}.btn.dark{background:#5d6873}.btn.danger{background:#c84646}
    input,select{width:100%;box-sizing:border-box;border:1px solid #dce3ea;border-radius:8px;padding:10px;margin-top:7px;background:#fff}
    label{display:block;color:#687483;font-size:14px;margin-top:12px}.result{margin-top:12px;padding:12px;border-radius:8px;background:#e8f0fb;color:#1f6fb2;font-weight:700}
    .list{display:grid;gap:8px;max-height:460px;overflow:auto}.item{border:1px solid #dce3ea;border-radius:8px;padding:10px;background:#fbfcfe}
    .item.warn{border-color:#d33b21;background:#fff4ef}
    @media(max-width:900px){.shell,.grid,.metrics{grid-template-columns:1fr}aside{position:relative}.top{display:block}}
  </style>
</head>
<body>
<div class="shell">
  <aside>
    <div class="brand">BusCheck Local</div>
    <div class="sub">Database/classes local</div>
    <div class="nav">
      <button onclick="startCamera()">Bật camera thêm học sinh</button>
      <button onclick="startScan()">Bắt đầu nhận diện</button>
      <button onclick="stopScan()">Dừng nhận diện</button>
      <button onclick="stopCamera()">Tắt camera</button>
      <button onclick="setMode('len')">Chế độ lên xe</button>
      <button onclick="setMode('xuong')">Chế độ xuống xe</button>
    </div>
  </aside>
  <main>
    <div class="top">
      <div><h1>Quan ly lop va diem danh</h1><p class="sub">Chon lop dang cho, he thong chi nhan dien hoc sinh trong lop do.</p></div>
      <strong id="clock"></strong>
    </div>
    <section class="metrics">
      <div class="metric"><span>Học sinh</span><strong id="mStudents">0</strong></div>
      <div class="metric"><span>Đã lên</span><strong id="mLen">0</strong></div>
      <div class="metric"><span>Đã xuống</span><strong id="mXuong">0</strong></div>
      <div class="metric"><span>Bất thường</span><strong id="mAnomaly">0</strong></div>
    </section>
    <section class="card" style="margin-bottom:16px">
      <h2>Danh sach lop tren xe</h2>
      <p class="sub">Moi chuyen xe chon mot lop dang cho. Nhan dien chi ap dung trong lop da chon.</p>
      <div class="row" id="classButtons" style="margin-top:12px"></div>
      <div class="result" id="activeClassResult">Chua chon lop dang cho.</div>
      <button class="btn danger" style="margin-top:12px" onclick="endTrip()">Ket thuc chuyen di</button>
      <div class="result" id="tripResult" style="display:none"></div>
    </section>
    <section class="grid">
      <div class="card">
        <h2>Camera</h2>
        <img class="stream" id="streamFrame" src="/frame.jpg" alt="camera frame">
        <div class="row">
          <button class="btn alt" onclick="startCamera()">Bật camera</button>
          <button class="btn" onclick="startScan()">Nhận diện</button>
          <button class="btn dark" onclick="stopScan()">Dừng nhận diện</button>
          <button class="btn" onclick="setMode('len')">Lên xe</button>
          <button class="btn" onclick="setMode('xuong')">Xuống xe</button>
        </div>
        <div class="result" id="lastResult">Đang tải...</div>
      </div>
      <div class="card">
        <h2>Them lop / hoc sinh</h2>
        <label>Tạo lớp mới<input id="newClassName" placeholder="7A1"></label>
        <button class="btn alt" onclick="createClass()">Tạo lớp</button>
        <label>Chon lop de them hoc sinh<select id="classSelect"></select></label>
        <label>Ho ten hoc sinh<input id="studentName" placeholder="Nguyen Van A"></label>
        <button class="btn alt" style="margin-top:12px" onclick="registerFace()">Chup mau & luu vao lop</button>
        <div class="result" id="registerResult">Bat camera, dung nhan dien, cho hoc sinh nhin thang roi luu mau.</div>
      </div>
    </section>
    <section class="grid" style="margin-top:16px">
      <div class="card">
        <h2>Nhat ky diem danh</h2>
        <div class="row" style="margin:10px 0 12px">
          <button class="btn danger" onclick="clearLogs()">Xoa nhat ky</button>
          <button class="btn danger" onclick="resetSystem()">Reset he thong</button>
        </div>
        <div class="list" id="records"></div>
      </div>
      <div class="card"><h2>Danh sach lop va hoc sinh</h2><div id="studentList" class="list"></div></div>
    </section>
  </main>
</div>
<script>
async function api(path, options){ const r = await fetch(path, options); return await r.json(); }
async function startCamera(){ await api('/api/start',{method:'POST'}); refresh(); }
async function stopCamera(){ await api('/api/stop',{method:'POST'}); refresh(); }
async function startScan(){ await api('/api/scan',{method:'POST',body:new URLSearchParams({enabled:'1'})}); refresh(); }
async function stopScan(){ await api('/api/scan',{method:'POST',body:new URLSearchParams({enabled:'0'})}); refresh(); }
async function setMode(mode){ await api('/api/mode',{method:'POST',body:new URLSearchParams({mode})}); refresh(); }
async function selectActiveClass(className){
  await api('/api/active-class',{method:'POST',body:new URLSearchParams({class_name: className})});
  refresh();
}
async function createClass(){
  const className = document.querySelector('#newClassName').value.trim();
  const data = await api('/api/classes',{method:'POST',body:new URLSearchParams({class_name: className})});
  document.querySelector('#registerResult').textContent = data.message;
  refresh();
}
async function registerFace(){
  const name = document.querySelector('#studentName').value.trim();
  const className = document.querySelector('#classSelect').value.trim();
  const data = await api('/api/register',{method:'POST',body:new URLSearchParams({name, class_name: className})});
  document.querySelector('#registerResult').textContent = data.message;
  refresh();
}
async function deleteStudent(name, className){
  await api('/api/student/delete',{method:'POST',body:new URLSearchParams({name, class_name: className})});
  refresh();
}
async function clearLogs(){
  if (!confirm('Xoa toan bo nhat ky diem danh va bat thuong?')) return;
  const data = await api('/api/logs/clear',{method:'POST'});
  document.querySelector('#lastResult').textContent = data.message;
  refresh();
}
async function resetSystem(){
  if (!confirm('Reset he thong se xoa lop, hoc sinh, mau khuon mat, anh diem danh va nhat ky. Ban chac chan?')) return;
  const data = await api('/api/system/reset',{method:'POST',body:new URLSearchParams({confirm:'RESET'})});
  document.querySelector('#lastResult').textContent = data.message;
  refresh();
}
async function endTrip(){
  const data = await api('/api/trip/end',{method:'POST'});
  const box = document.querySelector('#tripResult');
  box.style.display = 'block';
  if (!data.ok) {
    box.textContent = data.message;
    alert(data.message);
    return;
  }
  if (data.abnormal_count > 0) {
    const detail = data.abnormal.map(x=>`${x.name}: len ${x.len}, xuong ${x.xuong} - ${x.reason}`).join(String.fromCharCode(10));
    box.textContent = `${data.message}: ${data.abnormal_count} bat thuong`;
    alert(data.message + String.fromCharCode(10) + detail);
  } else {
    box.textContent = data.message;
    alert(data.message);
  }
  refresh();
}
function toggleClassBlock(id){
  const el = document.getElementById('students_' + id);
  if (el) el.hidden = !el.hidden;
}
function htmlEscape(value){
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function jsString(value){
  return encodeURIComponent(String(value));
}
function domId(value){
  return 'class_' + String(value).replace(/[^a-zA-Z0-9_-]/g, '_');
}
async function refresh(){
  const s = await api('/api/status');
  const r = await api('/api/report');
  const students = await api('/api/students');
  document.querySelector('#mStudents').textContent = students.students.length;
  document.querySelector('#mLen').textContent = r.len_count;
  document.querySelector('#mXuong').textContent = r.xuong_count;
  document.querySelector('#mAnomaly').textContent = r.anomaly_count;
  document.querySelector('#mAnomaly').closest('.metric').classList.toggle('warn', r.anomaly_count > 0);
  document.querySelector('#lastResult').textContent = `Mode ${s.mode.toUpperCase()} | Lop ${s.active_class || 'tat ca/chua chon'} | ${s.scan_enabled ? 'dang nhan dien' : 'camera chup mau'} | ${s.last_result.name} | score ${s.last_result.score} | ${s.last_result.message}`;
  document.querySelector('#activeClassResult').textContent = s.active_class ? `Lop dang cho: ${s.active_class}` : 'Chua chon lop dang cho.';
  document.querySelector('#classButtons').innerHTML = s.classes.map(c=>`<button class="btn ${c===s.active_class?'alt':''}" onclick="selectActiveClass(decodeURIComponent('${jsString(c)}'))">${htmlEscape(c)}</button>`).join('') || '<span class="sub">Chua co lop. Hay tao lop ben phai.</span>';
  const classSelect = document.querySelector('#classSelect');
  const currentClass = classSelect.value;
  classSelect.innerHTML = s.classes.map(c=>`<option value="${htmlEscape(c)}">${htmlEscape(c)}</option>`).join('') || '<option value="">Chua co lop</option>';
  if (s.classes.includes(currentClass)) classSelect.value = currentClass;
  document.querySelector('#records').innerHTML = r.records.map(x=>`<div class="item"><b>${htmlEscape(x.student_name)}</b> - ${htmlEscape(x.class_name || '')}<br>${x.mode} - ${x.confidence.toFixed(2)} - ${x.created_at}</div>`).join('') || '<div class="item">Chua co nhat ky</div>';
  document.querySelector('#studentList').innerHTML = students.classes.map(cls=>{
    const id = domId(cls.class_name);
    const rows = cls.students.map(x=>`<div class="item ${x.status === 'BAT_THUONG' ? 'warn' : ''}"><b>${htmlEscape(x.name)}</b><br>${x.templates} mau | Len ${x.len} / Xuong ${x.xuong} | ${x.status}<br><button class="btn danger" onclick="deleteStudent(decodeURIComponent('${jsString(x.name)}'), decodeURIComponent('${jsString(x.class_name)}'))">Xoa</button></div>`).join('');
    return `<div class="item"><button class="btn alt" onclick="toggleClassBlock('${id}')">${htmlEscape(cls.class_name)} - ${cls.count} hoc sinh</button><div id="students_${id}" style="margin-top:8px" hidden>${rows || '<div class="item">Lop nay chua co hoc sinh</div>'}</div></div>`;
  }).join('') || '<div class="item">Chua co lop/hoc sinh</div>';
}
let refreshBusy = false;
async function refreshSafe(){
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    await refresh();
  } catch (err) {
    document.querySelector('#lastResult').textContent = 'Loi tai du lieu: ' + err;
  } finally {
    refreshBusy = false;
  }
}
function refreshFrame(){
  document.querySelector('#streamFrame').src = '/frame.jpg?t=' + Date.now();
}
setInterval(()=>{document.querySelector('#clock').textContent = new Date().toLocaleString('vi-VN'); refreshSafe();}, 1500);
setInterval(refreshFrame, 250);
refreshSafe();
</script>
</body>
</html>"""


worker = CameraWorker()


def student_rows():
    conn = core.init_db()
    day = datetime.now().strftime("%Y-%m-%d")
    counts = {
        (row[0], row[1]): {"len": row[2] or 0, "xuong": row[3] or 0}
        for row in conn.execute(
            """
            SELECT student_name, class_name,
                   SUM(CASE WHEN mode='len' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN mode='xuong' THEN 1 ELSE 0 END)
            FROM attendance
            WHERE day=?
            GROUP BY student_name, class_name
            """,
            (day,),
        ).fetchall()
    }
    conn.close()

    rows = []
    seen = set()
    for info in core.load_students().values():
        key = (info["name"], info["class_name"])
        seen.add(key)
        c = counts.get(key, {"len": 0, "xuong": 0})
        rows.append({
            "name": info["name"],
            "class_name": info["class_name"],
            "templates": int(len(info["features"])),
            "len": c["len"],
            "xuong": c["xuong"],
            "status": "OK" if c["len"] == c["xuong"] else "BAT_THUONG",
        })

    for (name, class_name), c in counts.items():
        if (name, class_name) in seen:
            continue
        rows.append({
            "name": name,
            "class_name": class_name or "",
            "templates": 0,
            "len": c["len"],
            "xuong": c["xuong"],
            "status": "OK" if c["len"] == c["xuong"] else "BAT_THUONG",
        })
    return sorted(rows, key=lambda x: (x["class_name"], x["name"]))


def grouped_student_rows():
    groups = {class_name: [] for class_name in core.list_classes()}
    for row in student_rows():
        groups.setdefault(row["class_name"], []).append(row)
    return [
        {"class_name": class_name, "count": len(rows), "students": rows}
        for class_name, rows in sorted(groups.items())
    ]


def query_report():
    conn = core.init_db()
    day = datetime.now().strftime("%Y-%m-%d")
    records = conn.execute(
        "SELECT student_name, class_name, mode, confidence, created_at FROM attendance WHERE day=? ORDER BY id DESC LIMIT 30",
        (day,),
    ).fetchall()
    saved_anomaly_count = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
    conn.close()

    students = student_rows()
    len_count = sum(x["len"] for x in students)
    xuong_count = sum(x["xuong"] for x in students)
    mismatch_students = sum(1 for x in students if x["status"] == "BAT_THUONG")
    trip_gap = abs(len_count - xuong_count)
    return {
        "records": [
            {"student_name": r[0], "class_name": r[1], "mode": r[2], "confidence": float(r[3]), "created_at": r[4]}
            for r in records
        ],
        "students": students,
        "len_count": len_count,
        "xuong_count": xuong_count,
        "anomaly_count": max(mismatch_students, trip_gap),
        "mismatch_students": mismatch_students,
        "trip_gap": trip_gap,
        "saved_anomaly_count": saved_anomaly_count,
    }


def end_trip_report(class_name):
    class_name = class_name.strip()
    if not class_name:
        return {"ok": False, "message": "Hay chon lop dang cho truoc khi ket thuc chuyen"}

    rows = [row for row in student_rows() if row["class_name"] == class_name]
    abnormal = []
    for row in rows:
        if row["len"] == row["xuong"]:
            continue
        if row["len"] > row["xuong"]:
            reason = "hoc sinh da len nhung chua xuong"
        else:
            reason = "so lan xuong nhieu hon so lan len"
        abnormal.append({
            "name": row["name"],
            "class_name": row["class_name"],
            "len": row["len"],
            "xuong": row["xuong"],
            "reason": reason,
        })

    return {
        "ok": True,
        "class_name": class_name,
        "student_count": len(rows),
        "abnormal_count": len(abnormal),
        "abnormal": abnormal,
        "message": (
            f"Canh bao ket thuc chuyen lop {class_name}"
            if abnormal
            else f"Chuyen lop {class_name} OK: danh sach len/xuong khop nhau"
        ),
    }


def clear_logs():
    conn = core.init_db()
    conn.execute("DELETE FROM attendance")
    conn.execute("DELETE FROM anomalies")
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('attendance', 'anomalies')")
    conn.commit()
    conn.close()


def reset_system_data():
    clear_logs()
    if core.template_root().exists():
        shutil.rmtree(core.template_root())
    if core.PHOTO_DIR.exists():
        shutil.rmtree(core.PHOTO_DIR)
    core.ensure_dirs()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_text(HTML, "text/html; charset=utf-8")
        elif path == "/api/status":
            self.send_json(worker.snapshot())
        elif path == "/api/report":
            self.send_json(query_report())
        elif path == "/api/students":
            rows = student_rows()
            self.send_json({"students": rows, "classes": grouped_student_rows()})
        elif path == "/frame.jpg":
            self.frame_jpeg()
        elif path == "/stream":
            self.stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        data = parse_qs(body)

        if path == "/api/start":
            worker.start()
            self.send_json({"ok": True})
        elif path == "/api/stop":
            worker.stop()
            self.send_json({"ok": True})
        elif path == "/api/scan":
            worker.set_scan(data.get("enabled", ["0"])[0] == "1")
            self.send_json({"ok": True, "scan_enabled": worker.scan_enabled})
        elif path == "/api/mode":
            worker.set_mode(data.get("mode", ["len"])[0])
            self.send_json({"ok": True, "mode": worker.mode})
        elif path == "/api/active-class":
            worker.set_active_class(data.get("class_name", [""])[0])
            self.send_json({"ok": True, "active_class": worker.active_class})
        elif path == "/api/classes":
            class_name = data.get("class_name", [""])[0].strip()
            if not class_name:
                self.send_json({"ok": False, "message": "Hay nhap ten lop"})
                return
            core.create_class(class_name)
            self.send_json({"ok": True, "message": f"Da tao lop {class_name}", "classes": core.list_classes()})
        elif path == "/api/register":
            name = data.get("name", [""])[0].strip()
            class_name = data.get("class_name", [""])[0].strip()
            if not name:
                self.send_json({"ok": False, "message": "Hay nhap ten hoc sinh"})
                return
            ok, message = worker.register_current_face(name, class_name)
            self.send_json({"ok": ok, "message": message})
        elif path == "/api/student/delete":
            name = data.get("name", [""])[0].strip()
            class_name = data.get("class_name", [""])[0].strip()
            path_obj = core.student_file(name, class_name)
            if path_obj.exists():
                path_obj.unlink()
            worker.students = core.load_students()
            self.send_json({"ok": True, "message": "Da xoa hoc sinh"})
        elif path == "/api/trip/end":
            worker.set_scan(False)
            class_name = data.get("class_name", [worker.active_class])[0].strip()
            self.send_json(end_trip_report(class_name))
        elif path == "/api/logs/clear":
            clear_logs()
            worker.last_seen.clear()
            self.send_json({"ok": True, "message": "Da xoa nhat ky diem danh va bat thuong"})
        elif path == "/api/system/reset":
            if data.get("confirm", [""])[0] != "RESET":
                self.send_json({"ok": False, "message": "Can xac nhan reset"})
                return
            worker.stop()
            reset_system_data()
            worker.students = {}
            worker.active_class = ""
            worker.last_seen.clear()
            with worker.lock:
                worker.last_frame = None
                worker.last_jpeg = None
                worker.last_result = {
                    "name": "unknown",
                    "class_name": "",
                    "score": 0.0,
                    "message": "Da reset he thong",
                }
            self.send_json({"ok": True, "message": "Da reset he thong"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def stream(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            with worker.lock:
                jpeg = worker.last_jpeg
            if jpeg is None:
                time.sleep(0.2)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                break

    def frame_jpeg(self):
        with worker.lock:
            jpeg = worker.last_jpeg
        if jpeg is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (19, 32, 44)
            cv2.putText(frame, "Camera chua bat", (145, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 235, 245), 2)
            cv2.putText(frame, "Bam 'Bat camera' de xem hinh", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 180, 200), 2)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            jpeg = buffer.tobytes() if ok else b""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def send_json(self, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text, content_type):
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    core.ensure_dirs()
    core.init_db().close()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Pico2 web is running at http://{HOST}:{PORT}")
    print("Database templates: database/_pico2_classes/<class>/<student>.npz")
    server.serve_forever()


if __name__ == "__main__":
    main()
