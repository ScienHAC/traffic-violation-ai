# TrafficGuard AI — AI Traffic Violation Detection

AI system that detects traffic violations (no-helmet, triple-riding, no-seatbelt,
signal-jump, wrong-way) from live CCTV/IP-camera feeds or recorded video, logs
them with evidence, and shows them on a live dashboard.

## Architecture

```
                 ┌────────────────────────┐
 video file/RTSP │   api service :8000    │  two-stage detection:
 ───────────────▶│  (api/main.py)         │  1. YOLOv8s + ByteTrack tracks
                 │                        │     person/motorcycle/car/bus/truck
                 │  /ws/detect/video      │  2. Specialized models re-check
                 │  /ws/detect/live       │     crops: helmet, seatbelt
                 │  /detect/image         │  3. Geometry rules: triple-riding,
                 │  /alpr/read            │     signal-jump (stop-line ROI),
                 └───────────┬────────────┘     wrong-way (direction history)
                             │ POST confirmed violation (fire-and-forget)
                             ▼
                 ┌────────────────────────┐
                 │  judge service :8001   │  persists to SQLite, saves
                 │  (hybrid_mvp/server.py)│  evidence crop, broadcasts over
                 │  /api/violation        │  WebSocket, serves PDF reports
                 │  /api/violations       │
                 │  /api/*/report.pdf     │
                 └───────────┬────────────┘
                             │ WS + REST
                             ▼
                 ┌────────────────────────┐
                 │  frontend :5173         │  React/Vite dashboard:
                 │                        │  live feed, violation log,
                 │                        │  analytics, PDF downloads
                 └────────────────────────┘
```

## Violation types

| Type | Method | Model |
|---|---|---|
| `NO_HELMET` | crop rider's head, classify | [Viddesh1/Bike-Helmet-Detectionv2](https://github.com/Viddesh1/Bike-Helmet-Detectionv2) (YOLOv8, downloaded at build/first-run) |
| `TRIPLE_RIDING` | count person-boxes overlapping a tracked motorcycle | geometry only, no model |
| `NO_SEATBELT` | crop car windshield (top ~55% of bbox), classify | [RISEF/yolov11s-seatbelt](https://huggingface.co/RISEF/yolov11s-seatbelt) (YOLOv11-cls, downloaded at build/first-run) |
| `SIGNAL_JUMP` | tracked vehicle bbox overlaps a configured stop-line ROI | geometry only — pass `signal_roi=x1,y1,x2,y2` as a WS query param |
| `WRONG_WAY` | net centroid displacement over last 12 tracked frames opposes `expected_direction` | geometry only — pass `expected_direction=up\|down\|left\|right` |

No models were trained for this project — everything is a pretrained, publicly
licensed model wired into a shared two-stage detect→crop→classify pipeline
(`api/main.py::_two_stage_process_frame`). Check each linked repo/model's
license before commercial use.

## Measured performance

On the bundled sample clip (`hybrid_mvp/test_traffic.mp4`, 768×432): **~19 fps**
processed (every 3rd frame) on a CPU-only machine, no GPU. Helmet/seatbelt
classifiers ran without error and produced plausible per-vehicle confidences.
Accuracy (precision/recall) has **not** been measured against ground truth —
the pretrained weights come with their own reported metrics (see each model's
README linked above); treat detections as assistive, not enforcement-grade,
without further validation on your own footage.

## Run it

### Docker Compose (recommended — one command)

```bash
cp .env.example .env   # edit if deploying behind a public URL
docker compose up --build
```

- Dashboard: http://localhost:5173
- Detection API: http://localhost:8000/docs
- Judge/analytics API: http://localhost:8001/health

First build downloads ~150MB of model weights; subsequent starts are fast
(weights are cached in a named volume).

### Manual (dev)

```bash
# Backend — from trafficguard-prototype/
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
uv pip install -r requirements.txt -r hybrid_mvp/requirements.txt
uvicorn api.main:app --port 8000 &
uvicorn hybrid_mvp.server:app --port 8001 &

# Frontend
cd frontend && npm install && npm run dev
```

## Live camera / RTSP

Point the dashboard's camera input at an RTSP URL, or connect directly:

```
ws://localhost:8000/ws/detect/live?source=rtsp://user:pass@camera-ip/stream&signal_roi=0,300,768,400&expected_direction=down
```

Send `{"stop": true}` over the same socket to end the session. The identical
code path also powers `/ws/detect/video` for uploaded recordings — a camera
feed and a video file are both just a sequence of frames from OpenCV's
perspective, so nothing else differs.

## Reports & evidence

- Every confirmed violation gets a timestamped JPEG crop (`GET /static/violations/...`).
- Per-violation PDF: `GET /api/violations/{id}/report.pdf`
- Date-range summary PDF: `GET /api/reports/summary.pdf?start=...&end=...`

5-second evidence *video* clips (as opposed to a single crop) were scoped
but cut for time — the single timestamped image is the evidence artifact
for now. Add a rolling per-camera frame buffer + `cv2.VideoWriter` in
`api/main.py` if you need clips.

## Data & storage

SQLite (`hybrid_mvp/server.py`), not MySQL/Postgres — this is a single-VM
deployment with one writer process, so a file-based DB is the right amount
of infrastructure. It's accessed through SQLAlchemy, so swapping the
connection string is a one-line change if you ever need a networked DB.

## Deploying it publicly

Cheapest/fastest: run `docker compose up` on your machine and expose it with
a free Cloudflare Tunnel — no cloud bill:

```bash
cloudflared tunnel --url http://localhost:5173   # dashboard
cloudflared tunnel --url http://localhost:8000   # api (used by the dashboard)
cloudflared tunnel --url http://localhost:8001   # judge/analytics
```

Set `VITE_API_BASE` / `VITE_JUDGE_API_BASE` in `.env` to the api/judge tunnel
URLs before `docker compose up --build` so the built frontend calls the
public URLs instead of localhost.

For a persistent public URL instead: one GCP Compute Engine VM (e2-standard-4,
spot pricing), Docker Compose, same setup — see plan.md for details.

## Known limitations / what was cut

- No 5-second evidence clips (single image only) — see Reports & evidence above.
- Signal-jump and wrong-way are geometry rules (stop-line ROI / direction
  history), not real traffic-light-state or lane-polygon detection — you
  configure the zone/direction per camera via query params rather than
  drawing them in the UI.
- Seatbelt classifier's training set is small and imbalanced (see its model
  card) — expect it to need re-validation on your own footage.
- ALPR (EasyOCR) works but plate accuracy on low-res/angled Indian plates is
  unverified beyond a manual spot-check.
