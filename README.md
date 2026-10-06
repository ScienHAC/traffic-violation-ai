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
| `NO_HELMET` | a head-detector finds heads in the frame; a head only counts if it belongs to a rider sitting on a tracked **motorcycle** (pedal cycles and rickshaws are ignored). A rider is flagged after 2+ frames agree on "no helmet" (1 for a single photo) | [alexdjulin/BikeHelmetDetection](https://github.com/alexdjulin/BikeHelmetDetection) (YOLOv8n, MIT; trained on the CC0 Kaggle *Helmet Detection* set), downloaded at build/first-run |
| `TRIPLE_RIDING` | each person is assigned to the one bike they overlap most; 3+ riders on a bike in 2+ frames | geometry only, no model |
| `NO_SEATBELT` | crop car windshield (top ~55% of bbox), classify | [RISEF/yolov11s-seatbelt](https://huggingface.co/RISEF/yolov11s-seatbelt) (YOLOv11-cls, downloaded at build/first-run) |
| `SIGNAL_JUMP` | tracked vehicle bbox overlaps a configured stop-line ROI | geometry only — set in Settings (stop-line slider), or via WS params `check_signal=true&signal_roi=x1,y1,x2,y2` (values are % of the frame, 0-100) |
| `WRONG_WAY` | net centroid displacement over last 12 tracked frames opposes `expected_direction` | geometry only — pass `expected_direction=up\|down\|left\|right` |

No models were trained for this project — everything is a pretrained, publicly
licensed model wired into a shared two-stage detect→crop→classify pipeline
(`api/main.py::_two_stage_process_frame`). Check each linked repo/model's
license before commercial use.

## Measured performance

**Helmet detection** (checked by eye on real clips and on 200 labelled photos from the public
[Hirai-Labs helmet dataset](https://huggingface.co/datasets/Hirai-Labs/helmet-vlm-instruct-dataset)):

| | Before (crop the bike, classify) | Now (heads + rider check + frame voting) |
|---|---|---|
| Real rider with no helmet, rear view (Haridwar clip) | missed (0) | flagged |
| Helmeted riders on 5 other clips (Delhi, 3 × Barasat, Palembang) | 2 false alarms | 0 false alarms |
| Cyclist / pedal rickshaw | flagged | ignored |
| Labelled photos with a helmetless rider | ~4% caught, flagged every rider | ~45% caught, 0 false alarms on helmeted photos |

Recall is still the weak side: roughly half of helmetless riders in single photos are missed, and
from behind a black helmet and black hair look alike at CCTV distances. Treat it as assistive.

**Speed:**
On the bundled sample clip (`hybrid_mvp/test_traffic.mp4`, 768×432): **~19 fps**
processed (every 3rd frame) on a CPU-only machine, no GPU. Seatbelt, signal-jump and wrong-way
were checked on the Delhi clip. Seatbelt precision/recall is **not** measured against ground truth (see its
model card); treat all detections as assistive, not enforcement-grade, without validating on your own footage.

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
ws://localhost:8000/ws/detect/live?source=rtsp://user:pass@camera-ip/stream&check_signal=true&signal_roi=0,56,100,64&check_wrong_way=true&expected_direction=down
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
- Helmet model learned from ~760 public images, far smaller than commercial systems train on. The way
  to close the gap is fine-tuning on labelled Indian CCTV footage; the pipeline is ready for a better model
  (swap the weights file and `_HELMET_MODEL_URL`).
- Wrong-way needs motion, so it works on video only; images support helmet, triple-riding, seatbelt and
  signal-jump.
- Seatbelt classifier's training set is small and imbalanced (see its model
  card) — expect it to need re-validation on your own footage.
- ALPR (EasyOCR) works but plate accuracy on low-res/angled Indian plates is
  unverified beyond a manual spot-check.
