#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <WebServer.h>
#include <LittleFS.h>

// ============================================================
// Board config
// ============================================================
// IMPORTANT: change these pins to match your ESP32-S3-CAM board.
// The defaults below are common for ESP32-S3-EYE-like wiring, not universal.
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5
#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM    8
#define Y3_GPIO_NUM    9
#define Y2_GPIO_NUM    11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13

// Optional hardware mode switch pins. Pull to GND to select.
#define MODE_LEN_PIN   -1
#define MODE_XUONG_PIN -1

// Optional SIM7600 4G anomaly sender. Disabled by default.
#define USE_SIM7600 0
#define SIM7600_RX_PIN 44
#define SIM7600_TX_PIN 43
#define SIM7600_BAUD   115200
const char *SIM_APN = "internet";
const char *ALERT_HOST = "example.com";
const int ALERT_PORT = 80;
const char *ALERT_PATH = "/buscheck/anomaly";

// Wi-Fi AP mode. The device is fully local.
const char *AP_SSID = "BusCheck-S3";
const char *AP_PASS = "12345678";

// ============================================================
// Recognition profile
// ============================================================
// Camera is fixed; student face must be inside this crop.
// Frame is 320x240 grayscale. ROI defaults to centered 128x128.
static const int FRAME_W = 320;
static const int FRAME_H = 240;
static const int ROI_X = 96;
static const int ROI_Y = 56;
static const int ROI_W = 128;
static const int ROI_H = 128;

static const int FACE_W = 48;
static const int FACE_H = 48;
static const int GRID = 4;
static const int BINS = 16;
static const int FEATURE_DIM = GRID * GRID * BINS;
static const int MAX_STUDENTS = 64;
static const int MAX_TEMPLATES = 5;
static const int NAME_LEN = 32;
static const int CLASS_LEN = 12;
static const uint16_t DB_MAGIC = 0xB5C3;

// Score is cosine-like, scaled 0..1000. Tune after real samples.
static int MATCH_THRESHOLD = 720;
static const uint32_t REPEAT_GUARD_MS = 8000;

// ============================================================
// Data model
// ============================================================
struct Student {
  char name[NAME_LEN];
  char className[CLASS_LEN];
  uint8_t templateCount;
  uint8_t features[MAX_TEMPLATES][FEATURE_DIM];
};

struct Attendance {
  char name[NAME_LEN];
  char className[CLASS_LEN];
  char mode[8];
  int score;
  uint32_t ms;
};

Student students[MAX_STUDENTS];
uint16_t studentCount = 0;
Attendance lastLogs[20];
uint8_t logCount = 0;

String currentMode = "len";
String lastName = "unknown";
String lastClass = "";
int lastScore = 0;
uint32_t lastRecognizeMs = 0;
uint32_t frameCount = 0;
float fps = 0.0f;
uint32_t anomalyCount = 0;
String lastAnomaly = "";

uint32_t lastSeenLen[MAX_STUDENTS];
uint32_t lastSeenXuong[MAX_STUDENTS];
int boardCount[MAX_STUDENTS];
int offCount[MAX_STUDENTS];

WebServer server(80);

#if USE_SIM7600
HardwareSerial Sim7600(2);
#endif

String jsonEscape(const String &s);

// ============================================================
// Utilities
// ============================================================
void safeCopy(char *dst, size_t n, const String &src) {
  memset(dst, 0, n);
  src.substring(0, n - 1).toCharArray(dst, n);
}

void addLog(const Student &s, const char *mode, int score) {
  for (int i = 19; i > 0; --i) lastLogs[i] = lastLogs[i - 1];
  safeCopy(lastLogs[0].name, NAME_LEN, String(s.name));
  safeCopy(lastLogs[0].className, CLASS_LEN, String(s.className));
  safeCopy(lastLogs[0].mode, 8, String(mode));
  lastLogs[0].score = score;
  lastLogs[0].ms = millis();
  if (logCount < 20) logCount++;
}

#if USE_SIM7600
bool waitForModem(const char *expected, uint32_t timeoutMs) {
  uint32_t start = millis();
  String buffer;
  while (millis() - start < timeoutMs) {
    while (Sim7600.available()) {
      buffer += (char)Sim7600.read();
      if (buffer.indexOf(expected) >= 0) return true;
      if (buffer.indexOf("ERROR") >= 0) return false;
    }
    delay(10);
  }
  return false;
}

bool modemCmd(const String &cmd, const char *expected = "OK", uint32_t timeoutMs = 2000) {
  Sim7600.println(cmd);
  return waitForModem(expected, timeoutMs);
}

bool sendAnomaly4G(const String &payload) {
  modemCmd("AT", "OK", 1000);
  modemCmd("AT+HTTPTERM", "OK", 500);
  if (!modemCmd("AT+HTTPINIT")) return false;
  modemCmd("AT+HTTPPARA=\"CID\",1");
  modemCmd("AT+HTTPPARA=\"URL\",\"http://" + String(ALERT_HOST) + ":" + String(ALERT_PORT) + String(ALERT_PATH) + "\"");
  modemCmd("AT+HTTPPARA=\"CONTENT\",\"application/json\"");
  Sim7600.println("AT+HTTPDATA=" + String(payload.length()) + ",5000");
  if (!waitForModem("DOWNLOAD", 3000)) return false;
  Sim7600.print(payload);
  if (!waitForModem("OK", 6000)) return false;
  Sim7600.println("AT+HTTPACTION=1");
  bool ok = waitForModem("+HTTPACTION: 1,200", 15000);
  modemCmd("AT+HTTPTERM", "OK", 1000);
  return ok;
}
#endif

void reportAnomaly(const Student &s, const String &type, const String &message) {
  anomalyCount++;
  lastAnomaly = type + ": " + String(s.name);
  String payload = "{";
  payload += "\"bus_id\":\"BUS_01\",";
  payload += "\"student\":\"" + jsonEscape(String(s.name)) + "\",";
  payload += "\"class\":\"" + jsonEscape(String(s.className)) + "\",";
  payload += "\"type\":\"" + type + "\",";
  payload += "\"message\":\"" + jsonEscape(message) + "\",";
  payload += "\"ms\":" + String(millis());
  payload += "}";
  Serial.println("ANOMALY " + payload);
#if USE_SIM7600
  sendAnomaly4G(payload);
#endif
}

bool saveDatabase() {
  File f = LittleFS.open("/faces.db", "w");
  if (!f) return false;
  f.write((uint8_t *)&DB_MAGIC, sizeof(DB_MAGIC));
  f.write((uint8_t *)&studentCount, sizeof(studentCount));
  f.write((uint8_t *)students, sizeof(Student) * studentCount);
  f.close();
  return true;
}

bool loadDatabase() {
  if (!LittleFS.exists("/faces.db")) return false;
  File f = LittleFS.open("/faces.db", "r");
  if (!f) return false;
  uint16_t magic = 0;
  uint16_t count = 0;
  f.read((uint8_t *)&magic, sizeof(magic));
  f.read((uint8_t *)&count, sizeof(count));
  if (magic != DB_MAGIC || count > MAX_STUDENTS) {
    f.close();
    return false;
  }
  studentCount = count;
  f.read((uint8_t *)students, sizeof(Student) * studentCount);
  f.close();
  return true;
}

int findStudent(const String &name) {
  for (int i = 0; i < studentCount; ++i) {
    if (name == String(students[i].name)) return i;
  }
  return -1;
}

// ============================================================
// Camera + feature extraction
// ============================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;

  esp_err_t err = esp_camera_init(&config);
  return err == ESP_OK;
}

uint8_t sampleResized(camera_fb_t *fb, int x, int y) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= FRAME_W) x = FRAME_W - 1;
  if (y >= FRAME_H) y = FRAME_H - 1;
  return fb->buf[y * FRAME_W + x];
}

void extractFeature(camera_fb_t *fb, uint8_t out[FEATURE_DIM]) {
  uint16_t hist[FEATURE_DIM];
  memset(hist, 0, sizeof(hist));

  for (int yy = 1; yy < FACE_H - 1; ++yy) {
    for (int xx = 1; xx < FACE_W - 1; ++xx) {
      int srcX = ROI_X + (xx * ROI_W) / FACE_W;
      int srcY = ROI_Y + (yy * ROI_H) / FACE_H;
      uint8_t c = sampleResized(fb, srcX, srcY);

      uint8_t code = 0;
      code |= (sampleResized(fb, srcX - 2, srcY - 2) >= c) << 7;
      code |= (sampleResized(fb, srcX,     srcY - 2) >= c) << 6;
      code |= (sampleResized(fb, srcX + 2, srcY - 2) >= c) << 5;
      code |= (sampleResized(fb, srcX + 2, srcY)     >= c) << 4;
      code |= (sampleResized(fb, srcX + 2, srcY + 2) >= c) << 3;
      code |= (sampleResized(fb, srcX,     srcY + 2) >= c) << 2;
      code |= (sampleResized(fb, srcX - 2, srcY + 2) >= c) << 1;
      code |= (sampleResized(fb, srcX - 2, srcY)     >= c);

      int cellX = (xx * GRID) / FACE_W;
      int cellY = (yy * GRID) / FACE_H;
      int bin = code >> 4;
      int idx = (cellY * GRID + cellX) * BINS + bin;
      hist[idx]++;
    }
  }

  for (int cell = 0; cell < GRID * GRID; ++cell) {
    uint16_t sum = 0;
    for (int b = 0; b < BINS; ++b) sum += hist[cell * BINS + b];
    if (sum == 0) sum = 1;
    for (int b = 0; b < BINS; ++b) {
      out[cell * BINS + b] = (uint8_t)((hist[cell * BINS + b] * 255UL) / sum);
    }
  }
}

int featureScore(const uint8_t *a, const uint8_t *b) {
  uint32_t diff = 0;
  for (int i = 0; i < FEATURE_DIM; ++i) {
    diff += abs((int)a[i] - (int)b[i]);
  }
  int score = 1000 - (int)((diff * 1000UL) / (FEATURE_DIM * 255UL));
  if (score < 0) score = 0;
  return score;
}

int recognize(const uint8_t feature[FEATURE_DIM], int &bestScore) {
  int bestIndex = -1;
  bestScore = 0;
  for (int i = 0; i < studentCount; ++i) {
    for (int t = 0; t < students[i].templateCount; ++t) {
      int score = featureScore(feature, students[i].features[t]);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }
  }
  if (bestScore < MATCH_THRESHOLD) return -1;
  return bestIndex;
}

// ============================================================
// Attendance
// ============================================================
void updateModePins() {
  if (MODE_LEN_PIN >= 0 && digitalRead(MODE_LEN_PIN) == LOW) currentMode = "len";
  if (MODE_XUONG_PIN >= 0 && digitalRead(MODE_XUONG_PIN) == LOW) currentMode = "xuong";
}

void recordIfNeeded(int idx, int score) {
  uint32_t now = millis();
  bool isLen = currentMode == "len";
  uint32_t *lastSeen = isLen ? lastSeenLen : lastSeenXuong;
  if (now - lastSeen[idx] < REPEAT_GUARD_MS) return;

  if (isLen && boardCount[idx] > offCount[idx]) return;
  if (!isLen && offCount[idx] > boardCount[idx]) return;

  if (isLen) boardCount[idx]++;
  else offCount[idx]++;

  lastSeen[idx] = now;
  addLog(students[idx], currentMode.c_str(), score);

  if (!isLen && offCount[idx] > boardCount[idx]) {
    reportAnomaly(students[idx], "off_without_board", "Student got off without a matching board record");
  } else if (isLen && boardCount[idx] - offCount[idx] > 1) {
    reportAnomaly(students[idx], "double_board", "Student boarded again without getting off");
  }
}

void recognitionTask(void *param) {
  uint32_t lastFpsMs = millis();
  uint32_t fpsFrames = 0;
  while (true) {
    updateModePins();
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      delay(5);
      continue;
    }

    uint8_t feature[FEATURE_DIM];
    extractFeature(fb, feature);
    int score = 0;
    int idx = recognize(feature, score);
    lastScore = score;
    lastRecognizeMs = millis();
    if (idx >= 0) {
      lastName = String(students[idx].name);
      lastClass = String(students[idx].className);
      recordIfNeeded(idx, score);
    } else {
      lastName = "unknown";
      lastClass = "";
    }

    esp_camera_fb_return(fb);
    frameCount++;
    fpsFrames++;
    uint32_t now = millis();
    if (now - lastFpsMs >= 1000) {
      fps = fpsFrames * 1000.0f / (now - lastFpsMs);
      fpsFrames = 0;
      lastFpsMs = now;
    }
    delay(1);
  }
}

// ============================================================
// Web UI
// ============================================================
String jsonEscape(const String &s) {
  String out;
  for (int i = 0; i < s.length(); ++i) {
    char c = s[i];
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

void sendJsonStatus() {
  String json = "{";
  json += "\"mode\":\"" + currentMode + "\",";
  json += "\"lastName\":\"" + jsonEscape(lastName) + "\",";
  json += "\"lastClass\":\"" + jsonEscape(lastClass) + "\",";
  json += "\"lastScore\":" + String(lastScore) + ",";
  json += "\"fps\":" + String(fps, 1) + ",";
  json += "\"students\":" + String(studentCount) + ",";
  json += "\"anomalies\":" + String(anomalyCount) + ",";
  json += "\"lastAnomaly\":\"" + jsonEscape(lastAnomaly) + "\",";
  json += "\"logs\":[";
  for (int i = 0; i < logCount; ++i) {
    if (i) json += ",";
    json += "{";
    json += "\"name\":\"" + jsonEscape(String(lastLogs[i].name)) + "\",";
    json += "\"className\":\"" + jsonEscape(String(lastLogs[i].className)) + "\",";
    json += "\"mode\":\"" + jsonEscape(String(lastLogs[i].mode)) + "\",";
    json += "\"score\":" + String(lastLogs[i].score) + ",";
    json += "\"ms\":" + String(lastLogs[i].ms);
    json += "}";
  }
  json += "],\"studentList\":[";
  for (int i = 0; i < studentCount; ++i) {
    if (i) json += ",";
    json += "{";
    json += "\"name\":\"" + jsonEscape(String(students[i].name)) + "\",";
    json += "\"className\":\"" + jsonEscape(String(students[i].className)) + "\",";
    json += "\"templates\":" + String(students[i].templateCount) + ",";
    json += "\"len\":" + String(boardCount[i]) + ",";
    json += "\"xuong\":" + String(offCount[i]);
    json += "}";
  }
  json += "]}";
  server.send(200, "application/json", json);
}

void handleRegister() {
  String name = server.arg("name");
  String className = server.arg("className");
  if (!name.length()) {
    server.send(400, "text/plain", "Missing name");
    return;
  }

  int idx = findStudent(name);
  if (idx < 0) {
    if (studentCount >= MAX_STUDENTS) {
      server.send(400, "text/plain", "Student database full");
      return;
    }
    idx = studentCount++;
    memset(&students[idx], 0, sizeof(Student));
    safeCopy(students[idx].name, NAME_LEN, name);
    safeCopy(students[idx].className, CLASS_LEN, className);
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  uint8_t feature[FEATURE_DIM];
  extractFeature(fb, feature);
  esp_camera_fb_return(fb);

  Student &s = students[idx];
  if (s.templateCount < MAX_TEMPLATES) {
    memcpy(s.features[s.templateCount], feature, FEATURE_DIM);
    s.templateCount++;
  } else {
    for (int i = 1; i < MAX_TEMPLATES; ++i) memcpy(s.features[i - 1], s.features[i], FEATURE_DIM);
    memcpy(s.features[MAX_TEMPLATES - 1], feature, FEATURE_DIM);
  }
  saveDatabase();
  server.send(200, "text/plain", "Saved template for " + name);
}

void handleDelete() {
  String name = server.arg("name");
  int idx = findStudent(name);
  if (idx < 0) {
    server.send(404, "text/plain", "Not found");
    return;
  }
  for (int i = idx + 1; i < studentCount; ++i) {
    students[i - 1] = students[i];
    boardCount[i - 1] = boardCount[i];
    offCount[i - 1] = offCount[i];
  }
  studentCount--;
  saveDatabase();
  server.send(200, "text/plain", "Deleted " + name);
}

void handleMode() {
  String mode = server.arg("mode");
  if (mode == "len" || mode == "xuong") currentMode = mode;
  server.send(200, "text/plain", currentMode);
}

void handleResetDay() {
  memset(boardCount, 0, sizeof(boardCount));
  memset(offCount, 0, sizeof(offCount));
  logCount = 0;
  server.send(200, "text/plain", "Day counters reset");
}

void handleJpeg() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "capture failed");
    return;
  }

  uint8_t *jpgBuf = NULL;
  size_t jpgLen = 0;
  bool ok = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, fb->format, 70, &jpgBuf, &jpgLen);
  esp_camera_fb_return(fb);

  if (!ok || jpgBuf == NULL || jpgLen == 0) {
    server.send(500, "text/plain", "jpeg encode failed");
    return;
  }

  server.setContentLength(jpgLen);
  server.send(200, "image/jpeg", "");
  WiFiClient client = server.client();
  client.write(jpgBuf, jpgLen);
  free(jpgBuf);
}

const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BusCheck ESP32-S3</title>
<style>
body{margin:0;background:#eef2f6;color:#17212b;font-family:Segoe UI,Arial,sans-serif}
.shell{display:grid;grid-template-columns:260px 1fr;min-height:100vh}.side{background:#132433;color:#fff;padding:22px}
h1,h2,p{margin:0}.sub{color:#9fb2c1;margin-top:5px}.nav{display:grid;gap:9px;margin-top:24px}
button{border:0;border-radius:8px;padding:11px 13px;background:#1f6fb2;color:#fff;cursor:pointer;font:inherit}
button.alt{background:#177d87}button.dark{background:#5d6873}button.danger{background:#c84646}
main{padding:24px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
.card,.metric{background:#fff;border:1px solid #dce3ea;border-radius:8px;box-shadow:0 16px 40px #1f2e3c18}
.metric{padding:15px}.metric span{color:#687483}.metric b{font-size:28px;display:block;margin-top:5px}
.grid{display:grid;grid-template-columns:minmax(360px,1.4fr) minmax(320px,.8fr);gap:16px}.card{padding:16px}
img{width:100%;aspect-ratio:4/3;object-fit:cover;background:#14202b;border-radius:8px;margin:12px 0}
input{width:100%;box-sizing:border-box;border:1px solid #dce3ea;border-radius:8px;padding:10px;margin:7px 0 11px}
.item{border:1px solid #dce3ea;background:#fbfcfe;border-radius:8px;padding:10px;margin-top:8px}
@media(max-width:900px){.shell,.grid,.metrics{grid-template-columns:1fr}}
</style></head><body><div class="shell"><aside class="side"><h1>BusCheck S3</h1><p class="sub">Local recognition on ESP32-S3-CAM</p>
<div class="nav"><button onclick="mode('len')">Lên xe</button><button onclick="mode('xuong')">Xuống xe</button><button class="danger" onclick="resetDay()">Reset ngày</button></div>
</aside><main><h1>Điểm danh học sinh trên xe</h1><p class="sub">Camera cố định, vùng mặt cố định, nhận diện LBP local.</p>
<section class="metrics"><div class="metric"><span>FPS</span><b id="fps">0</b></div><div class="metric"><span>Học sinh</span><b id="students">0</b></div><div class="metric"><span>Mode</span><b id="mode">len</b></div><div class="metric"><span>Bất thường</span><b id="anomalies">0</b></div></section>
<section class="grid"><div class="card"><h2>Camera</h2><img id="cam" src="/jpg"><button onclick="reloadCam()">Refresh ảnh</button><div class="item"><b id="last">unknown</b><br><span id="lastClass"></span></div></div>
<div class="card"><h2>Thêm khuôn mặt</h2><label>Tên</label><input id="name"><label>Lớp</label><input id="className"><button class="alt" onclick="register()">Chụp vùng mặt & lưu template</button><div class="item" id="msg">Cho học sinh đứng đúng vị trí rồi lưu 5 mẫu.</div></div></section>
<section class="grid" style="margin-top:16px"><div class="card"><h2>Nhật ký</h2><div id="logs"></div></div><div class="card"><h2>Danh sách học sinh</h2><div id="list"></div></div></section>
</main></div><script>
async function api(p,o){let r=await fetch(p,o);return await r.text()}
async function status(){let r=await fetch('/status');let s=await r.json();fps.textContent=s.fps;students.textContent=s.students;modeEl=document.getElementById('mode');modeEl.textContent=s.mode;anomalies.textContent=s.anomalies;last.textContent=s.lastName+' score '+s.lastScore;lastClass.textContent=(s.lastClass||'')+' '+(s.lastAnomaly?'| '+s.lastAnomaly:'');
logs.innerHTML=s.logs.map(x=>`<div class=item><b>${x.name}</b> ${x.mode} score ${x.score}<br>${Math.round(x.ms/1000)}s</div>`).join('')||'<div class=item>Chưa có</div>';
list.innerHTML=s.studentList.map(x=>`<div class=item><b>${x.name}</b> ${x.className}<br>Templates ${x.templates} | Lên ${x.len} / Xuống ${x.xuong}<br><button class=danger onclick="del('${x.name}')">Xóa</button></div>`).join('')||'<div class=item>Chưa có học sinh</div>'}
async function mode(m){await api('/mode?mode='+m);status()}async function resetDay(){await api('/reset-day');status()}
async function register(){let n=name.value,c=className.value;msg.textContent=await api('/register?name='+encodeURIComponent(n)+'&className='+encodeURIComponent(c));status()}
async function del(n){await api('/delete?name='+encodeURIComponent(n));status()}function reloadCam(){cam.src='/jpg?t='+Date.now()}
setInterval(()=>{status();reloadCam()},1000);status();
</script></body></html>
)HTML";

void handleIndex() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void setupWeb() {
  server.on("/", HTTP_GET, handleIndex);
  server.on("/status", HTTP_GET, sendJsonStatus);
  server.on("/register", HTTP_GET, handleRegister);
  server.on("/delete", HTTP_GET, handleDelete);
  server.on("/mode", HTTP_GET, handleMode);
  server.on("/reset-day", HTTP_GET, handleResetDay);
  server.on("/jpg", HTTP_GET, handleJpeg);
  server.begin();
}

// ============================================================
// Setup / loop
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  if (MODE_LEN_PIN >= 0) pinMode(MODE_LEN_PIN, INPUT_PULLUP);
  if (MODE_XUONG_PIN >= 0) pinMode(MODE_XUONG_PIN, INPUT_PULLUP);

#if USE_SIM7600
  Sim7600.begin(SIM7600_BAUD, SERIAL_8N1, SIM7600_RX_PIN, SIM7600_TX_PIN);
  modemCmd("AT", "OK", 1000);
  modemCmd("AT+CGDCONT=1,\"IP\",\"" + String(SIM_APN) + "\"", "OK", 2000);
#endif

  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS failed");
  }
  loadDatabase();

  memset(lastSeenLen, 0, sizeof(lastSeenLen));
  memset(lastSeenXuong, 0, sizeof(lastSeenXuong));
  memset(boardCount, 0, sizeof(boardCount));
  memset(offCount, 0, sizeof(offCount));

  if (!initCamera()) {
    Serial.println("Camera init failed. Check pin config.");
    while (true) delay(1000);
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());

  setupWeb();
  xTaskCreatePinnedToCore(recognitionTask, "recognition", 8192, NULL, 1, NULL, 1);
}

void loop() {
  server.handleClient();
  delay(1);
}
