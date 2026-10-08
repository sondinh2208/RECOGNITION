const API_BASE = "http://localhost:8000";
const STORAGE_KEY = "buscheck_state_v2";
const MAX_PHOTOS = 10;
const SCAN_INTERVAL_MS = 1400;
const REPEAT_GUARD_MS = 9000;

const state = loadState();
let activeMode = "len";
let cameraStream = null;
let scanTimer = null;
let scanBusy = false;
let lastRecognizedAt = {};

const els = {
    tabs: document.querySelectorAll(".nav-item"),
    panels: document.querySelectorAll(".tab-panel"),
    pageTitle: document.getElementById("pageTitle"),
    todayLabel: document.getElementById("todayLabel"),
    timeLabel: document.getElementById("timeLabel"),
    studentCount: document.getElementById("studentCount"),
    boardCount: document.getElementById("boardCount"),
    offCount: document.getElementById("offCount"),
    issueCount: document.getElementById("issueCount"),
    apiStatusText: document.getElementById("apiStatusText"),
    apiStatusDot: document.getElementById("apiStatusDot"),
    modeBtns: document.querySelectorAll(".mode-btn"),
    video: document.getElementById("cameraStream"),
    canvas: document.getElementById("captureCanvas"),
    cameraEmpty: document.getElementById("cameraEmpty"),
    startCamera: document.getElementById("startCamera"),
    stopCamera: document.getElementById("stopCamera"),
    scanStatus: document.getElementById("scanStatus"),
    recognitionName: document.getElementById("recognitionName"),
    recognitionMeta: document.getElementById("recognitionMeta"),
    activityList: document.getElementById("activityList"),
    studentForm: document.getElementById("studentForm"),
    studentName: document.getElementById("studentName"),
    studentClass: document.getElementById("studentClass"),
    studentPhoto: document.getElementById("studentPhoto"),
    studentList: document.getElementById("studentList"),
    compareTable: document.getElementById("compareTable"),
    reportList: document.getElementById("reportList"),
    photoAudit: document.getElementById("photoAudit"),
    clearData: document.getElementById("clearData"),
    toast: document.getElementById("toast"),
};

document.addEventListener("DOMContentLoaded", async () => {
    bindEvents();
    tickClock();
    setInterval(tickClock, 1000);
    renderAll();
    await checkApi();
    await syncStudentsFromApi();
});

function loadState() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
        return JSON.parse(saved);
    }
    return { students: [], records: [] };
}

function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function bindEvents() {
    els.tabs.forEach((tab) => {
        tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });

    els.modeBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            activeMode = btn.dataset.mode;
            els.modeBtns.forEach((item) => item.classList.remove("active"));
            btn.classList.add("active");
            updateScanStatus();
        });
    });

    els.startCamera.addEventListener("click", startCamera);
    els.stopCamera.addEventListener("click", stopCamera);
    els.studentForm.addEventListener("submit", addStudent);
    els.clearData.addEventListener("click", clearLocalHistory);
}

function switchTab(tabId) {
    const titles = {
        attendance: "Điểm danh tự động lên/xuống xe",
        students: "Embedding khuôn mặt học sinh",
        compare: "So sánh học sinh lên/xuống",
        report: "Báo cáo bất thường",
    };

    els.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
    els.panels.forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
    els.pageTitle.textContent = titles[tabId];
    renderAll();
}

function tickClock() {
    const now = new Date();
    els.todayLabel.textContent = now.toLocaleDateString("vi-VN", {
        weekday: "long",
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
    });
    els.timeLabel.textContent = now.toLocaleTimeString("vi-VN");
}

async function checkApi() {
    try {
        const response = await fetch(`${API_BASE}/`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        els.apiStatusDot.classList.add("online");
        els.apiStatusText.textContent = `API nhận diện đang chạy (${data.so_nguoi_da_dang_ky || 0} khuôn mặt)`;
        return true;
    } catch (error) {
        els.apiStatusDot.classList.remove("online");
        els.apiStatusText.textContent = "Chưa kết nối API nhận diện";
        showToast("Hãy chạy python api.py trước khi nhận diện", true);
        return false;
    }
}

async function syncStudentsFromApi() {
    try {
        const response = await fetch(`${API_BASE}/nguoi-dung`);
        const data = await response.json();
        const names = Array.isArray(data.danh_sach) ? data.danh_sach : [];

        names.forEach((name) => {
            if (!state.students.some((student) => student.name === name)) {
                state.students.push({
                    id: crypto.randomUUID(),
                    name,
                    className: "Chưa nhập",
                    photos: [],
                    embedded: true,
                });
            }
        });

        saveState();
        renderAll();
    } catch {
        renderAll();
    }
}

async function startCamera() {
    const apiReady = await checkApi();
    if (!apiReady) {
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 } });
        els.video.srcObject = cameraStream;
        els.video.classList.add("active");
        els.cameraEmpty.hidden = true;
        startAutoScan();
        showToast("Camera đã bật, hệ thống bắt đầu tự nhận diện");
    } catch (error) {
        showToast(`Không mở được camera: ${error.message}`, true);
    }
}

function stopCamera() {
    stopAutoScan();
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }
    els.video.srcObject = null;
    els.video.classList.remove("active");
    els.cameraEmpty.hidden = false;
    updateScanStatus();
}

function startAutoScan() {
    stopAutoScan();
    updateScanStatus();
    scanTimer = window.setInterval(scanAndRecognize, SCAN_INTERVAL_MS);
    scanAndRecognize();
}

function stopAutoScan() {
    if (scanTimer) {
        window.clearInterval(scanTimer);
        scanTimer = null;
    }
    scanBusy = false;
}

function updateScanStatus(message) {
    if (message) {
        els.scanStatus.textContent = message;
        return;
    }
    if (!cameraStream) {
        els.scanStatus.textContent = "Camera tắt, chưa quét nhận diện.";
        return;
    }
    els.scanStatus.textContent = `Đang tự quét ở chế độ ${activeMode === "len" ? "Lên xe" : "Xuống xe"}.`;
}

function captureFrameDataUrl() {
    if (!cameraStream || !els.video.videoWidth) {
        return null;
    }
    els.canvas.width = els.video.videoWidth;
    els.canvas.height = els.video.videoHeight;
    els.canvas.getContext("2d").drawImage(els.video, 0, 0);
    return els.canvas.toDataURL("image/jpeg", 0.86);
}

async function scanAndRecognize() {
    if (scanBusy || !cameraStream) {
        return;
    }

    const photo = captureFrameDataUrl();
    if (!photo) {
        return;
    }

    scanBusy = true;
    updateScanStatus("Đang gửi frame sang model nhận diện...");

    try {
        const formData = new FormData();
        formData.append("anh", dataUrlToBlob(photo), "camera-frame.jpg");
        formData.append("tu_dong_mo_cua", "false");

        const response = await fetch(`${API_BASE}/nhan-dien`, {
            method: "POST",
            body: formData,
        });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Lỗi nhận diện");
        }

        if (result.da_nhan_ra && result.ten && result.ten !== "khong_xac_dinh") {
            const confidence = Math.round((result.do_chinh_xac || 0) * 100);
            els.recognitionName.textContent = result.ten;
            els.recognitionMeta.textContent = `Độ khớp ${confidence}% - chế độ ${activeMode === "len" ? "Lên xe" : "Xuống xe"}`;
            handleRecognizedStudent(result.ten, confidence, photo);
        } else {
            els.recognitionName.textContent = "Chưa nhận ra học sinh";
            els.recognitionMeta.textContent = "Đưa khuôn mặt rõ hơn vào khung camera.";
            updateScanStatus(`Đang tự quét ở chế độ ${activeMode === "len" ? "Lên xe" : "Xuống xe"}.`);
        }
    } catch (error) {
        updateScanStatus("Lỗi nhận diện, hệ thống sẽ thử lại.");
        showToast(error.message, true);
    } finally {
        scanBusy = false;
    }
}

function handleRecognizedStudent(name, confidence, photo) {
    const student = ensureStudent(name);
    const now = Date.now();
    const guardKey = `${activeMode}:${student.name}`;
    const lastAt = lastRecognizedAt[guardKey] || 0;

    if (now - lastAt < REPEAT_GUARD_MS) {
        updateScanStatus(`Đã nhận ra ${student.name}, bỏ qua để tránh ghi lặp.`);
        return;
    }

    const status = getStudentTripStatus(student.id);
    if (activeMode === "len" && status.boardCount > status.offCount) {
        updateScanStatus(`${student.name} đã được ghi lên xe, chờ chuyển chế độ xuống xe.`);
        return;
    }
    if (activeMode === "xuong" && status.offCount > status.boardCount) {
        updateScanStatus(`${student.name} đã được ghi xuống xe, bỏ qua lượt lặp.`);
        return;
    }

    lastRecognizedAt[guardKey] = now;
    addAttendanceRecord(student, activeMode, confidence, photo);
    updateScanStatus(`Tự ghi nhận ${student.name} ${activeMode === "len" ? "lên xe" : "xuống xe"}.`);
}

function ensureStudent(name) {
    let student = state.students.find((item) => item.name === name);
    if (!student) {
        student = {
            id: crypto.randomUUID(),
            name,
            className: "Chưa nhập",
            photos: [],
            embedded: true,
        };
        state.students.push(student);
    }
    return student;
}

function addAttendanceRecord(student, mode, confidence, photo) {
    student.photos.push({
        src: photo,
        capturedAt: new Date().toISOString(),
        mode,
    });
    if (student.photos.length > MAX_PHOTOS) {
        student.photos = student.photos.slice(student.photos.length - MAX_PHOTOS);
    }

    const now = new Date();
    state.records.push({
        id: crypto.randomUUID(),
        studentId: student.id,
        name: student.name,
        className: student.className,
        mode,
        confidence,
        at: now.toISOString(),
        date: formatDateKey(now),
        time: now.toLocaleTimeString("vi-VN"),
        photo,
    });

    saveState();
    renderAll();
    showToast(`${student.name} đã ${mode === "len" ? "lên xe" : "xuống xe"}`);
}

async function addStudent(event) {
    event.preventDefault();
    const name = els.studentName.value.trim();
    const className = els.studentClass.value.trim();
    const file = els.studentPhoto.files[0];

    if (!name || !className || !file) {
        showToast("Cần nhập họ tên, lớp và chọn ảnh mặt học sinh", true);
        return;
    }

    try {
        const formData = new FormData();
        formData.append("ten", name);
        formData.append("anh", file);

        const response = await fetch(`${API_BASE}/dang-ky`, {
            method: "POST",
            body: formData,
        });
        const result = await response.json();

        if (!response.ok || !result.thanh_cong) {
            throw new Error(result.detail || result.thong_bao || "Không embedding được khuôn mặt");
        }

        const photo = await readFileAsDataUrl(file);
        let student = state.students.find((item) => item.name === name);
        if (!student) {
            student = { id: crypto.randomUUID(), name, className, photos: [], embedded: true };
            state.students.push(student);
        }
        student.className = className;
        student.embedded = true;
        student.photos = [{ src: photo, capturedAt: new Date().toISOString(), mode: "dang_ky" }];

        els.studentForm.reset();
        saveState();
        renderAll();
        await checkApi();
        showToast(`Đã embedding và lưu khuôn mặt: ${name}`);
    } catch (error) {
        showToast(error.message, true);
    }
}

function readFileAsDataUrl(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target.result);
        reader.readAsDataURL(file);
    });
}

function clearLocalHistory() {
    if (!confirm("Xóa lịch sử điểm danh và ảnh chụp trên trình duyệt này? Embedding .npy trong API vẫn được giữ.")) {
        return;
    }
    state.records = [];
    state.students.forEach((student) => {
        student.photos = student.photos.filter((photo) => photo.mode === "dang_ky").slice(-1);
    });
    saveState();
    renderAll();
}

function renderAll() {
    renderMetrics();
    renderActivities();
    renderStudents();
    renderCompare();
    renderReports();
    renderPhotoAudit();
}

function renderMetrics() {
    const todayRecords = getTodayRecords();
    const board = todayRecords.filter((r) => r.mode === "len").length;
    const off = todayRecords.filter((r) => r.mode === "xuong").length;
    const issues = getComparison().filter((row) => row.status !== "Khớp").length;

    els.studentCount.textContent = state.students.length;
    els.boardCount.textContent = board;
    els.offCount.textContent = off;
    els.issueCount.textContent = issues;
}

function renderActivities() {
    const records = [...state.records].reverse().slice(0, 10);
    els.activityList.innerHTML = records.length
        ? records.map((record) => `
            <article class="activity-item">
                <img src="${record.photo}" alt="">
                <div>
                    <strong>${escapeHtml(record.name)}</strong>
                    <span>${record.mode === "len" ? "Lên xe" : "Xuống xe"} - ${formatDateTime(record.at)} - ${record.confidence || 0}%</span>
                </div>
                <b class="${record.mode}">${record.mode === "len" ? "Lên" : "Xuống"}</b>
            </article>
        `).join("")
        : `<div class="empty-state">Chưa có lượt điểm danh nào.</div>`;
}

function renderStudents() {
    els.studentList.innerHTML = state.students.length
        ? state.students.map((student) => {
            const latest = student.photos[student.photos.length - 1];
            return `
                <article class="student-card">
                    <img src="${latest ? latest.src : makePlaceholderPhoto(student.name)}" alt="">
                    <div class="student-info">
                        <strong>${escapeHtml(student.name)}</strong>
                        <span>${escapeHtml(student.className || "Chưa nhập")} - ${student.photos.length}/10 ảnh - ${student.embedded ? "Đã embedding" : "Chưa embedding"}</span>
                    </div>
                    <button class="icon-btn" title="Xóa học sinh" onclick="deleteStudent('${student.id}')">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </article>
            `;
        }).join("")
        : `<div class="empty-state">Chưa có học sinh. Hãy thêm ảnh mặt để tạo embedding.</div>`;
}

function renderCompare() {
    const rows = getComparison();
    els.compareTable.innerHTML = rows.length
        ? rows.map((row) => `
            <tr>
                <td>${escapeHtml(row.name)}</td>
                <td>${escapeHtml(row.className)}</td>
                <td>${row.lastBoard ? formatDateTime(row.lastBoard) : "--"}</td>
                <td>${row.lastOff ? formatDateTime(row.lastOff) : "--"}</td>
                <td>${row.boardCount}/${row.offCount}</td>
                <td><span class="badge ${row.status === "Khớp" ? "ok" : "warn"}">${row.status}</span></td>
            </tr>
        `).join("")
        : `<tr><td colspan="6" class="empty-cell">Chưa có học sinh để so sánh.</td></tr>`;
}

function renderReports() {
    const issues = getComparison().filter((row) => row.status !== "Khớp");
    els.reportList.innerHTML = issues.length
        ? issues.map((row) => `
            <article class="issue-item">
                <i class="fa-solid fa-circle-exclamation"></i>
                <div>
                    <strong>${escapeHtml(row.name)}</strong>
                    <span>${row.status}. Lên: ${row.boardCount}, xuống: ${row.offCount}.</span>
                </div>
            </article>
        `).join("")
        : `<div class="success-state"><i class="fa-solid fa-circle-check"></i><span>Không có bất thường trong hôm nay.</span></div>`;
}

function renderPhotoAudit() {
    els.photoAudit.innerHTML = state.students.length
        ? state.students.map((student) => `
            <article class="audit-row">
                <div>
                    <strong>${escapeHtml(student.name)}</strong>
                    <span>${student.photos.length}/10 ảnh đã lưu</span>
                </div>
                <div class="photo-strip">
                    ${student.photos.map((photo, index) => `<img src="${photo.src}" title="Ảnh ${index + 1}" alt="Ảnh ${index + 1}">`).join("")}
                </div>
            </article>
        `).join("")
        : `<div class="empty-state">Chưa có ảnh học sinh.</div>`;
}

function getTodayRecords() {
    const today = formatDateKey(new Date());
    return state.records.filter((record) => record.date === today);
}

function getStudentTripStatus(studentId) {
    const records = getTodayRecords().filter((record) => record.studentId === studentId);
    const board = records.filter((record) => record.mode === "len");
    const off = records.filter((record) => record.mode === "xuong");
    return {
        boardCount: board.length,
        offCount: off.length,
        lastBoard: board.at(-1)?.at,
        lastOff: off.at(-1)?.at,
    };
}

function getComparison() {
    return state.students.map((student) => {
        const status = getStudentTripStatus(student.id);
        let label = "Khớp";
        if (status.boardCount > status.offCount) {
            label = "Đã lên, chưa xuống";
        } else if (status.offCount > status.boardCount) {
            label = "Xuống nhiều hơn lên";
        }

        return {
            name: student.name,
            className: student.className || "Chưa nhập",
            ...status,
            status: label,
        };
    });
}

async function deleteStudent(id) {
    const student = state.students.find((item) => item.id === id);
    if (!student || !confirm(`Xóa học sinh ${student.name}?`)) {
        return;
    }

    try {
        await fetch(`${API_BASE}/nguoi-dung/${encodeURIComponent(student.name)}`, { method: "DELETE" });
    } catch {
        // Van xoa o giao dien neu API dang tat.
    }

    state.students = state.students.filter((item) => item.id !== id);
    state.records = state.records.filter((item) => item.studentId !== id);
    saveState();
    renderAll();
    await checkApi();
}

function dataUrlToBlob(dataUrl) {
    const [meta, payload] = dataUrl.split(",");
    const mime = meta.match(/:(.*?);/)?.[1] || "image/jpeg";
    const binary = atob(payload);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: mime });
}

function makePlaceholderPhoto(name) {
    const initials = name.split(" ").filter(Boolean).slice(-2).map((word) => word[0]).join("").toUpperCase();
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#eef2f8";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#1e5b6f";
    ctx.beginPath();
    ctx.arc(160, 92, 42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#2f8a9f";
    ctx.beginPath();
    ctx.roundRect(92, 145, 136, 68, 28);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 42px Segoe UI";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(initials || "HS", 160, 170);
    return canvas.toDataURL("image/png");
}

function formatDateKey(date) {
    return date.toISOString().slice(0, 10);
}

function formatDateTime(value) {
    return new Date(value).toLocaleString("vi-VN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        day: "2-digit",
        month: "2-digit",
    });
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
    }[char]));
}

function showToast(message, isError = false) {
    els.toast.textContent = message;
    els.toast.classList.toggle("error", isError);
    els.toast.classList.add("show");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => els.toast.classList.remove("show"), 3000);
}
