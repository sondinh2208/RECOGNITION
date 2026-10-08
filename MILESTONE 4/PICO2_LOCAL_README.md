# Pico2-style local attendance

This is the ultra-light local version. It does not use InsightFace, FastAPI, a
server, or the web dashboard. It is designed like a constrained edge device:
low resolution camera frames, Haar face detection, compact local features, local
SQLite logs, rolling 10 photos per student, and optional anomaly-only webhook.

## Commands

Register a student:

```bash
python pico2_local.py register --name "Nguyen Van A" --class-name "7A1"
```

During registration:

- Press `S` to save one face template.
- Save 5-10 templates per student with small head movements.
- Press `Q` to finish.

Run attendance:

```bash
python pico2_local.py run --mode len
```

Keyboard while running:

- `L`: switch to on-board mode.
- `X`: switch to off-board mode.
- `Q`: quit.

Show local report:

```bash
python pico2_local.py report
```

Run local web UI:

```bash
python pico2_web.py
```

Open:

```text
http://127.0.0.1:8088
```

The web UI can:

- Start/stop camera.
- Switch on-board/off-board mode.
- Show MJPEG camera stream.
- Create classes.
- Register a student into a selected class from the current camera frame.
- Keep registration/capture separate from recognition/attendance.
- Show local attendance, student list, and anomaly report.

Self test:

```bash
python pico2_local.py self-test
```

## Local data

- Existing InsightFace database remains in `database/*.npy`.
- Pico2-style class/student templates: `database/_pico2_classes/<class>/<student>.npz`
- Attendance log: `pico2_attendance.db`
- Rolling photos: `pico2_photos/<student>/`
- Config: `pico2_config.json`

## Anomaly-only 4G sending

Keep `alert_webhook_url` empty for fully local mode. If you add a webhook URL,
the program only sends anomaly JSON, not normal logs, photos, or embeddings.

## Speed profile

The default run mode uses:

- Camera: 320x240
- Processing: 160x120
- Face size: 48x48
- Detection every 3 frames
- Recognition every 2 frames

This is intentionally much lighter than InsightFace. It is the closest software
shape to a Pico2-class device, but real Pico2 hardware still cannot match modern
InsightFace accuracy.
