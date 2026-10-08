# BusCheck ESP32-S3-CAM

Local face attendance firmware for ESP32-S3-CAM with PSRAM.

This version is designed for:

- fixed camera position
- fixed student face position
- local-only recognition
- web UI served directly by ESP32
- student face template management
- target recognition loop above 15 FPS on ESP32-S3-CAM with PSRAM

It does **not** use InsightFace. It uses a tiny LBP histogram template matcher
over a fixed ROI. This is much lighter and can run on ESP32-S3, but it cannot
match modern deep face recognition accuracy.

## Files

- `esp32_s3_cam_buscheck.ino`: Arduino firmware

## Arduino IDE Setup

Install:

- ESP32 Arduino core

Board suggestion:

- ESP32S3 Dev Module
- PSRAM: OPI PSRAM / Enabled
- Flash: 8MB or 16MB if available
- Partition: any partition with LittleFS/SPIFFS space

## Pin Configuration

Open `esp32_s3_cam_buscheck.ino` and change this block for your camera board:

```cpp
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5
...
```

ESP32-S3-CAM boards do not all share the same camera pins. If Serial shows:

```text
Camera init failed. Check pin config.
```

the pin map is wrong for your board.

## Web UI

After flashing, connect your phone/laptop to Wi-Fi:

```text
SSID: BusCheck-S3
PASS: 12345678
```

Open:

```text
http://192.168.4.1
```

The web UI supports:

- switch `Len xe` / `Xuong xe`
- view latest camera frame
- register face template from the fixed crop area
- list students
- delete students
- show attendance logs
- reset daily counters

## Registration Workflow

1. Put the student face in the fixed camera position.
2. Enter student name and class.
3. Click `Chup vung mat & luu template`.
4. Repeat 5 times per student with small face variation.

The firmware keeps up to 5 templates per student.

## Fixed ROI

The firmware assumes the face is in this area of the 320x240 frame:

```cpp
static const int ROI_X = 96;
static const int ROI_Y = 56;
static const int ROI_W = 128;
static const int ROI_H = 128;
```

Adjust these values after mounting the camera.

## Accuracy Tuning

The default threshold:

```cpp
static int MATCH_THRESHOLD = 720;
```

Increase it to reduce false matches.
Decrease it if real students are often not recognized.

Recommended starting range:

```text
680 - 780
```

## Performance Notes

To keep FPS high:

- Camera runs grayscale QVGA: 320x240.
- No face detection is performed.
- Fixed ROI is resized logically to 48x48.
- LBP feature is 256 bytes.
- Matching is integer L1-distance style.
- Web `/jpg` endpoint is a snapshot, not full MJPEG stream, to avoid slowing recognition.

## 4G Anomaly Sending

This firmware currently provides the local core and web UI. For SIM7600 4G,
add a UART task that sends only anomaly JSON when:

- student gets off without boarding
- student boards twice without getting off
- unknown/low-score attempts exceed a threshold

Suggested JSON:

```json
{
  "bus_id": "BUS_01",
  "student": "Nguyen Van A",
  "type": "off_without_board",
  "time_ms": 123456
}
```

Normal images/templates should stay local.
