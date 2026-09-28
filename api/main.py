"""
api/main.py — TrafficGuard FastAPI Backend
  Wraps the existing src/ detection pipeline behind REST + WebSocket endpoints.
  Models are lazy-loaded as singletons (expensive YOLO / PaddleOCR weights
  only load on first request, not at import time).

Usage:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
import os
# Ensure project root is on sys.path so `from src.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import base64
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TrafficGuard — Traffic Violation Detection API",
    version="1.0.0",
    description="YOLOv8 + PaddleOCR violation detection REST and WebSocket API",
)

_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:5174",   # Vite fallback port
    "http://localhost:3000",   # CRA / alt dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
]
_frontend_url = os.getenv("FRONTEND_URL")
if _frontend_url:
    _origins.append(_frontend_url)
if os.getenv("CORS_ALLOW_ALL") == "true":
    _origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── GPU auto-detection ───────────────────────────────────────────────────────

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

print(f"[api/main.py] Compute device: {_DEVICE.upper()}")

# ─── Two-stage helmet pipeline constants ──────────────────────────────────────

_HELMET_MODEL_URL   = "https://raw.githubusercontent.com/Viddesh1/Bike-Helmet-Detectionv2/main/weights/best.pt"
_HELMET_MODEL_LOCAL = Path(__file__).parent.parent / "trafficguard-prototype" / "models" / "bike_helmet_yolov8.pt"
# Normalise path so it works whether called from repo root or api/ dir
if not _HELMET_MODEL_LOCAL.parent.exists():
    _HELMET_MODEL_LOCAL = Path(__file__).parent.parent / "models" / "bike_helmet_yolov8.pt"
_HELMET_MODEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)

# Seatbelt classifier — YOLOv11s-cls, binary (no_seatbelt / seat_belt), run on
# the top half (windshield) of a car bbox. https://huggingface.co/RISEF/yolov11s-seatbelt
_SEATBELT_MODEL_URL   = "https://huggingface.co/RISEF/yolov11s-seatbelt/resolve/main/weights/best.pt"
_SEATBELT_MODEL_LOCAL = _HELMET_MODEL_LOCAL.parent / "seatbelt_yolov11s_cls.pt"
_SEATBELT_CONF = 0.55

_HELMET_CONF = 0.65
_COCO_CONF   = 0.30
_MIN_VEH_PX  = 40
_HEAD_PAD    = 0.35   # 35% extra upward padding on crop

# COCO class IDs
_CLS_PERSON = 0; _CLS_BICYCLE = 1; _CLS_CAR = 2
_CLS_MOTO   = 3; _CLS_BUS     = 5; _CLS_TRUCK = 7
_BIKE_CLASSES  = {_CLS_MOTO, _CLS_BICYCLE}
_MOTOR_CLASSES = {_CLS_CAR, _CLS_BUS, _CLS_TRUCK}
_ALL_CLASSES   = [_CLS_PERSON, _CLS_BICYCLE, _CLS_CAR, _CLS_MOTO, _CLS_BUS, _CLS_TRUCK]
_CLS_LABEL     = {0:"PERSON",1:"BICYCLE",2:"CAR",3:"MOTO",5:"BUS",7:"TRUCK"}

_COLORS = {
    "NO_HELMET":     (0,  80,  255),
    "TRIPLE_RIDING": (0,  0,   255),
    "NO_SEATBELT":   (0,  140, 255),
    "SIGNAL_JUMP":   (0,  215, 255),
    "WRONG_WAY":     (180,0,   255),
    "OK":            (0,  200, 0),
    "SCAN":          (160,160, 0),
    "SMALL":         (60, 60,  60),
}
_VIOLATION_TAGS = {
    "NO_HELMET":     "NO HELMET ⚠",
    "TRIPLE_RIDING": "TRIPLE! ⚠",
    "NO_SEATBELT":   "NO SEATBELT ⚠",
    "SIGNAL_JUMP":   "SIGNAL JUMP ⚠",
    "WRONG_WAY":     "WRONG WAY ⚠",
}

JUDGE_SERVER = os.getenv("JUDGE_SERVER", "http://localhost:8001/api/violation")

# ─── Lazy-loaded singletons ───────────────────────────────────────────────────

_detector       = None
_alpr           = None
_coco_model     = None   # YOLO COCO model for two-stage pipeline
_helmet_model   = None   # helmet YOLO model
_seatbelt_model = None   # seatbelt YOLO classifier
_no_helmet_ids  : set = set()
_helmet_ids     : set = set()


def get_detector(conf: float = 0.45) -> Any:
    """Return the shared detector singleton, loading on first call."""
    global _detector
    if _detector is None:
        from src.detect import TrafficGuardDetector
        _detector = TrafficGuardDetector(conf_threshold=conf)
    return _detector


def get_alpr() -> Any:
    """Return the shared ALPR singleton, loading on first call."""
    global _alpr
    if _alpr is None:
        try:
            from src.alpr import ALPRPipeline
            _alpr = ALPRPipeline()
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"ALPR deps missing (pip install paddleocr paddlepaddle): {e}",
            )
    return _alpr


def get_checker(
    overlap: float = 0.30,
    triple: int = 3,
    signal_roi: Optional[List[int]] = None,
) -> Any:
    """ViolationChecker is lightweight — recreate per request with user params."""
    from src.violations import ViolationChecker
    return ViolationChecker(
        overlap_threshold=overlap,
        triple_riding_threshold=triple,
        signal_roi=signal_roi,
    )


# ─── Two-stage pipeline helpers ───────────────────────────────────────────────

def _ensure_helmet_weights() -> Optional[Path]:
    """Download helmet weights once and cache; reuse on later runs."""
    if _HELMET_MODEL_LOCAL.exists() and _HELMET_MODEL_LOCAL.stat().st_size > 1_000_000:
        return _HELMET_MODEL_LOCAL
    print(f"[api] Downloading helmet weights (~88MB) from:\n      {_HELMET_MODEL_URL}")
    try:
        import requests as req
        with req.get(_HELMET_MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = _HELMET_MODEL_LOCAL.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(_HELMET_MODEL_LOCAL)
        print(f"[api] Helmet weights saved → {_HELMET_MODEL_LOCAL}")
        return _HELMET_MODEL_LOCAL
    except Exception as e:
        print(f"[api][WARN] Could not download helmet weights: {e}")
        return None


def _get_coco_model():
    """Singleton YOLO COCO model for the two-stage pipeline."""
    global _coco_model
    if _coco_model is None:
        from ultralytics import YOLO
        print("[api] Loading COCO tracking model (yolov8s.pt)…")
        _coco_model = YOLO("yolov8s.pt")
        _coco_model.to(_DEVICE)
        print(f"[api] COCO model ready on {_DEVICE.upper()}")
    return _coco_model


def _get_helmet_model():
    """Singleton helmet YOLO model."""
    global _helmet_model, _no_helmet_ids, _helmet_ids
    if _helmet_model is None:
        weights = _ensure_helmet_weights()
        if weights is None:
            print("[api][WARN] No helmet weights — helmet checks will be skipped.")
            return None
        try:
            from ultralytics import YOLO
            print(f"[api] Loading helmet model: {weights}")
            _helmet_model = YOLO(str(weights))
            _helmet_model.to(_DEVICE)
            _no_helmet_ids = {
                cid for cid, name in _helmet_model.names.items()
                if any(kw in name.lower() for kw in ["no_helmet","no-helmet","without","bare"])
            }
            _helmet_ids = {
                cid for cid, name in _helmet_model.names.items()
                if "helmet" in name.lower() and cid not in _no_helmet_ids
            }
            print(f"[api] Helmet IDs={_helmet_ids}  No-helmet IDs={_no_helmet_ids}")
        except Exception as e:
            print(f"[api][WARN] Could not load helmet model: {e}")
            _helmet_model = None
    return _helmet_model


def _ensure_seatbelt_weights() -> Optional[Path]:
    """Download seatbelt classifier weights once and cache; reuse on later runs."""
    if _SEATBELT_MODEL_LOCAL.exists() and _SEATBELT_MODEL_LOCAL.stat().st_size > 1_000_000:
        return _SEATBELT_MODEL_LOCAL
    print(f"[api] Downloading seatbelt weights (~11MB) from:\n      {_SEATBELT_MODEL_URL}")
    try:
        import requests as req
        with req.get(_SEATBELT_MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = _SEATBELT_MODEL_LOCAL.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(_SEATBELT_MODEL_LOCAL)
        print(f"[api] Seatbelt weights saved → {_SEATBELT_MODEL_LOCAL}")
        return _SEATBELT_MODEL_LOCAL
    except Exception as e:
        print(f"[api][WARN] Could not download seatbelt weights: {e}")
        return None


def _get_seatbelt_model():
    """Singleton seatbelt YOLO classifier."""
    global _seatbelt_model
    if _seatbelt_model is None:
        weights = _ensure_seatbelt_weights()
        if weights is None:
            print("[api][WARN] No seatbelt weights — seatbelt checks will be skipped.")
            return None
        try:
            from ultralytics import YOLO
            print(f"[api] Loading seatbelt model: {weights}")
            _seatbelt_model = YOLO(str(weights))
            _seatbelt_model.to(_DEVICE)
        except Exception as e:
            print(f"[api][WARN] Could not load seatbelt model: {e}")
            _seatbelt_model = None
    return _seatbelt_model


def _crop_windshield(frame: np.ndarray, box: list) -> np.ndarray:
    """Top ~55% of a car bbox — where the windshield / driver sits."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bh = y2 - y1
    x1 = max(0, int(x1) - 10)
    y1 = max(0, int(y1) - 10)
    x2 = min(w, int(x2) + 10)
    y2 = min(h, int(y1 + bh * 0.55))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame[:10, :10].copy()
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 224:
        scale = 224 / max(min(ch, cw), 1)
        crop = cv2.resize(crop, (max(int(cw*scale), 224), max(int(ch*scale), 224)),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop.copy()


def _check_seatbelt(crop_img: np.ndarray) -> str:
    """Returns 'NO_SEATBELT', 'SEATBELT', or 'SKIP'."""
    smodel = _get_seatbelt_model()
    if smodel is None:
        return "SKIP"
    try:
        results = smodel.predict(crop_img, verbose=False, imgsz=224)
        if not results or results[0].probs is None:
            return "SKIP"
        probs = results[0].probs
        top1, top1conf = int(probs.top1), float(probs.top1conf)
        if top1conf < _SEATBELT_CONF:
            return "SKIP"
        name = results[0].names[top1].lower()
        return "NO_SEATBELT" if "no" in name else "SEATBELT"
    except Exception as e:
        print(f"  [SEATBELT ERR] {e}")
        return "SKIP"


def _check_signal_jump(box: list, signal_roi: Optional[list]) -> bool:
    """True if bbox overlaps the configured red-light stop-line ROI."""
    if not signal_roi:
        return False
    rx1, ry1, rx2, ry2 = signal_roi
    return _boxes_overlap(box, [rx1, ry1, rx2, ry2])


def _check_wrong_way(history: list, expected_direction: Optional[str]) -> bool:
    """
    True if a vehicle's net centroid displacement over its tracked history
    opposes the configured expected direction of travel.
    `expected_direction` is one of "up" | "down" | "left" | "right".
    """
    if not expected_direction or len(history) < 6:
        return False
    (x0, y0), (x1, y1) = history[0], history[-1]
    dx, dy = x1 - x0, y1 - y0
    _WRONG_WAY_PX = 25   # ignore jitter below this net displacement
    if expected_direction == "down":
        return dy < -_WRONG_WAY_PX
    if expected_direction == "up":
        return dy > _WRONG_WAY_PX
    if expected_direction == "right":
        return dx < -_WRONG_WAY_PX
    if expected_direction == "left":
        return dx > _WRONG_WAY_PX
    return False


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _crop_head(frame: np.ndarray, box: list) -> np.ndarray:
    h, w = frame.shape[:2]
    bh = box[3] - box[1]
    top_ex = int(bh * _HEAD_PAD)
    x1 = max(0, int(box[0]) - 20)
    y1 = max(0, int(box[1]) - 20 - top_ex)
    x2 = min(w, int(box[2]) + 20)
    y2 = min(h, int(box[3]) + 20)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame[:10, :10].copy()
    # Upscale small crops
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 224:
        scale = 224 / min(ch, cw)
        crop = cv2.resize(crop, (max(int(cw*scale), 224), max(int(ch*scale), 224)),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop.copy()


def _check_helmet(crop_img: np.ndarray) -> str:
    """Returns 'NO_HELMET', 'HELMET', or 'SKIP'."""
    hmodel = _get_helmet_model()
    if hmodel is None:
        return "SKIP"
    try:
        results = hmodel.predict(crop_img, conf=_HELMET_CONF, verbose=False, imgsz=320)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return "SKIP"
        clses = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        relevant = [(c, cf) for c, cf in zip(clses, confs)
                    if c in _helmet_ids or c in _no_helmet_ids]
        if not relevant:
            return "SKIP"
        best = max(relevant, key=lambda x: x[1])[0]
        return "NO_HELMET" if best in _no_helmet_ids else "HELMET"
    except Exception as e:
        print(f"  [HELMET ERR] {e}")
        return "SKIP"


def _encode_jpeg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def _annotate(
    frame: np.ndarray,
    bikes: list,
    persons: list,
    motor_vehs: list,
    violations: list,
    known_violations: Optional[dict] = None,
) -> np.ndarray:
    """
    Draw bounding boxes + labels on frame.
    `known_violations` is a persistent dict {vehicle_id: violation_type} that
    accumulates across all frames so previously confirmed vehicles keep their
    highlight even on frames where they emit no new violation event.
    """
    disp = frame.copy()
    # Merge this frame's new violations into the running tally
    if known_violations is not None:
        for v in violations:
            known_violations[v["vehicle_id"]] = v["violation_type"]
    # Build per-frame lookup — use known_violations when available
    viol_types = known_violations if known_violations is not None else \
                 {v["vehicle_id"]: v["violation_type"] for v in violations}

    for p in persons:
        x1,y1,x2,y2 = [int(v) for v in p["box"]]
        cv2.rectangle(disp, (x1,y1), (x2,y2), (190,190,0), 1)

    for veh in motor_vehs + bikes:
        x1,y1,x2,y2 = [int(v) for v in veh["box"]]
        vid = veh["id"]
        lbl = _CLS_LABEL.get(veh["cls"], "VEH")
        vtype = viol_types.get(vid, "")
        if vtype in _COLORS:
            color = _COLORS[vtype]
            tag   = f"{lbl}#{vid} {_VIOLATION_TAGS[vtype]}"
            cv2.rectangle(disp, (x1-2,y1-2), (x2+2,y2+2), color, 3)
        else:
            color = _COLORS["OK"] if veh in bikes else _COLORS["SCAN"]
            tag   = f"{lbl}#{vid}"
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
        cv2.putText(disp, tag, (x1+1, max(15, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3)
        cv2.putText(disp, tag, (x1, max(14, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
    return disp


async def _post_to_judge(vehicle_id: int, violation_type: str,
                          crop_bytes: bytes, frame_id: int) -> None:
    """Fire-and-forget POST to the Judge Feed server on port 8001."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                JUDGE_SERVER,
                data={"vehicle_id": str(vehicle_id),
                      "violation_type": violation_type,
                      "frame_id": str(frame_id)},
                files={"image": ("crop.jpg", crop_bytes, "image/jpeg")},
            )
    except Exception:
        pass   # judge server offline — ignore silently


def _two_stage_process_frame(
    frame: np.ndarray,
    frame_id: int,
    model,
    confirmed_ids: set,
    known_violations: Optional[dict] = None,
    signal_roi: Optional[list] = None,
    expected_direction: Optional[str] = None,
    track_history: Optional[dict] = None,
) -> tuple:
    """
    Run the two-stage detection on a single frame.
    Returns (annotated_frame, violations_this_frame).

    `confirmed_ids` is a shared set to avoid re-flagging the same vehicle.
    `known_violations` is a shared dict {vehicle_id: violation_type} that persists
    across ALL frames so previously confirmed vehicles keep their color box even
    when they produce no new violation in the current frame.
    `signal_roi` [x1,y1,x2,y2] flags any vehicle crossing a red-light stop line.
    `expected_direction` ("up"|"down"|"left"|"right") flags vehicles moving
    against traffic; `track_history` {vehicle_id: [(cx,cy), ...]} backs it.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _mk_violation(vtype: str, veh: dict, conf: Optional[float] = None, rider_count: int = 1) -> dict:
        try:
            cb = _encode_jpeg(_crop_head(frame, veh["box"]))
        except Exception:
            cb = b""
        return {
            "violation_type": vtype,
            "vehicle_id": veh["id"],
            "bbox": veh["box"],
            "confidence": round(conf if conf is not None else veh["conf"], 3),
            "frame_id": frame_id,
            "timestamp": ts,
            "plate_text": "UNKNOWN",
            "rider_count": rider_count,
            "_crop_bytes": cb,
        }

    # ── COCO tracking ────────────────────────────────────────────────────
    try:
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=_COCO_CONF, classes=_ALL_CLASSES,
            device=_DEVICE, verbose=False, imgsz=640, iou=0.45,
        )
    except Exception as e:
        print(f"[YOLO ERR] {e}")
        return frame.copy(), []

    persons   : list = []
    bikes     : list = []
    motor_vehs: list = []
    violations: list = []   # NEW violations detected in this frame only

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        res    = results[0]
        bboxes = res.boxes.xyxy.cpu().numpy()
        clses  = res.boxes.cls.cpu().numpy().astype(int)
        confs  = res.boxes.conf.cpu().numpy()
        ids    = (res.boxes.id.cpu().numpy().astype(int)
                  if res.boxes.id is not None
                  else np.full(len(clses), -1, dtype=int))

        for i in range(len(clses)):
            box = bboxes[i].tolist()
            det = {"box": box, "conf": float(confs[i]),
                   "id": int(ids[i]), "cls": int(clses[i]),
                   "bw": box[2]-box[0], "bh": box[3]-box[1]}
            c = det["cls"]
            if   c == _CLS_PERSON:    persons.append(det)
            elif c in _BIKE_CLASSES:  bikes.append(det)
            elif c in _MOTOR_CLASSES: motor_vehs.append(det)

    # ── Bike violation checks ─────────────────────────────────────────────
    for veh in bikes:
        vid, box = veh["id"], veh["box"]
        bw, bh   = veh["bw"], veh["bh"]

        if bw < _MIN_VEH_PX or bh < _MIN_VEH_PX or vid < 0:
            continue

        riders = [p for p in persons if _boxes_overlap(p["box"], box)]
        count  = len(riders)

        # Triple riding — geometry only
        if count >= 3:
            key = (vid, "TRIPLE_RIDING")
            if key not in confirmed_ids:
                confirmed_ids.add(key)
                avg_conf = sum(r["conf"] for r in riders) / len(riders)
                violations.append(_mk_violation("TRIPLE_RIDING", veh, conf=avg_conf, rider_count=count))
            continue

        # Helmet check — YOLO second stage (once per vehicle ID)
        key = (vid, "HELMET_CHECK")
        if key not in confirmed_ids:
            confirmed_ids.add(key)
            crop_img = _crop_head(frame, box)
            verdict  = _check_helmet(crop_img)
            if verdict == "NO_HELMET":
                vkey = (vid, "NO_HELMET")
                if vkey not in confirmed_ids:
                    confirmed_ids.add(vkey)
                    violations.append(_mk_violation("NO_HELMET", veh, rider_count=count))

    # ── Track centroid history (feeds wrong-way direction check) ──────────
    if track_history is not None:
        for veh in bikes + motor_vehs:
            cx = (veh["box"][0] + veh["box"][2]) / 2
            cy = (veh["box"][1] + veh["box"][3]) / 2
            hist = track_history.setdefault(veh["id"], [])
            hist.append((cx, cy))
            if len(hist) > 12:
                del hist[:-12]

    # ── Signal jump + wrong-way — apply to every tracked bike/vehicle ─────
    for veh in bikes + motor_vehs:
        vid, box = veh["id"], veh["box"]
        if vid < 0:
            continue
        if _check_signal_jump(box, signal_roi):
            key = (vid, "SIGNAL_JUMP")
            if key not in confirmed_ids:
                confirmed_ids.add(key)
                violations.append(_mk_violation("SIGNAL_JUMP", veh))
        hist = (track_history or {}).get(vid, [])
        if _check_wrong_way(hist, expected_direction):
            key = (vid, "WRONG_WAY")
            if key not in confirmed_ids:
                confirmed_ids.add(key)
                violations.append(_mk_violation("WRONG_WAY", veh))

    # ── Seatbelt check — cars only, once per vehicle ID ────────────────────
    for veh in motor_vehs:
        if veh["cls"] != _CLS_CAR or veh["id"] < 0:
            continue
        key = (veh["id"], "SEATBELT_CHECK")
        if key not in confirmed_ids:
            confirmed_ids.add(key)
            if _check_seatbelt(_crop_windshield(frame, veh["box"])) == "NO_SEATBELT":
                vkey = (veh["id"], "NO_SEATBELT")
                if vkey not in confirmed_ids:
                    confirmed_ids.add(vkey)
                    violations.append(_mk_violation("NO_SEATBELT", veh))

    # Draw with persistent known_violations so confirmed boxes stay colored
    annotated = _annotate(frame, bikes, persons, motor_vehs, violations, known_violations)
    # Return detection lists so the caller can cache them for skipped-frame annotation
    return annotated, violations, bikes, persons, motor_vehs


# ─── Pydantic response models ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


class DetectionOut(BaseModel):
    cls: str
    conf: float
    bbox: List[float]
    track_id: int


class ViolationOut(BaseModel):
    violation_type: str
    vehicle_id: int
    bbox: List[float]
    confidence: float
    frame_id: int
    timestamp: str
    plate_text: str
    rider_count: int
    detection_tier: Optional[str] = None
    ocr_confidence_raw: Optional[float] = None
    plate_valid: bool = False


class DetectImageResponse(BaseModel):
    detections: List[DetectionOut]
    violations: List[ViolationOut]
    annotated_image_b64: str = Field(
        description="JPEG annotated frame as base64 — use as <img src='data:image/jpeg;base64,...'>"
    )
    frame_id: int = 0
    timestamp: str


class ALPRResponse(BaseModel):
    raw_text: str
    plate_number: str
    confidence: float
    detection_tier: Optional[str] = None
    ocr_confidence_raw: Optional[float] = None
    plate_valid: bool = False


class ViolationRecord(BaseModel):
    violation_id: str = ""
    plate_number: str = ""
    violation_type: str = ""
    confidence: float = 0.0
    timestamp: str = ""
    frame_id: Any = 0
    plate_valid: bool = False
    image_path: str = ""


class ViolationLogResponse(BaseModel):
    count: int
    records: List[Dict[str, Any]]


class ViolationSummaryResponse(BaseModel):
    total: int
    NO_HELMET: int
    TRIPLE_RIDING: int
    SIGNAL_JUMP: int
    NO_SEATBELT: int = 0
    WRONG_WAY: int = 0


class DatasetFetchRoboflowRequest(BaseModel):
    api_key: str
    workspace: str
    project: str
    version: int
    format: str = "yolov8"


class DatasetFetchResponse(BaseModel):
    status: str
    message: str
    path: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to BGR numpy image, raising 400 on failure."""
    arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image — ensure it's a valid JPEG/PNG/BMP.")
    return frame


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame to JPEG base64 string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _detection_to_dict(d) -> dict:
    return {
        "cls": d.cls,
        "conf": round(float(d.conf), 4),
        "bbox": [round(float(x), 2) for x in d.bbox],
        "track_id": int(d.track_id),
    }


def _violation_to_dict(v) -> dict:
    return {
        "violation_type": v.violation_type,
        "vehicle_id": int(v.vehicle_id),
        "bbox": [round(float(x), 2) for x in v.bbox],
        "confidence": round(float(v.confidence), 4),
        "frame_id": int(v.frame_id),
        "timestamp": v.timestamp,
        "plate_text": getattr(v, "plate_text", "UNKNOWN"),
        "rider_count": int(v.rider_count),
        "detection_tier": getattr(v, "detection_tier", None),
        "ocr_confidence_raw": getattr(v, "ocr_confidence_raw", None),
        "plate_valid": getattr(v, "plate_valid", False),
    }


def _run_alpr_on_violation(frame: np.ndarray, violation, alpr):
    """Detect plate within a vehicle bbox and return PlateResult or None."""
    try:
        # Support both object and dict formats for bbox
        bbox = getattr(violation, "bbox", None) or violation.get("bbox")
        plate_bboxes, tier = alpr.detect_plates_in_vehicle(frame, bbox)
        if plate_bboxes:
            result = alpr.read_plate_from_frame(frame, plate_bboxes[0])
            if result:
                result.detection_tier = tier
                return result
    except Exception:
        pass
    return None


# ─── Endpoints ────────────────────────────────────────────────────────────────

# GET /health
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check."""
    return {"status": "ok"}


# POST /detect/image
@app.post("/detect/image", response_model=DetectImageResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(..., description="Traffic image (JPEG/PNG/BMP)"),
    conf_threshold: float = Query(0.45, ge=0.1, le=0.9, description="YOLO confidence threshold"),
    overlap_threshold: float = Query(0.30, ge=0.1, le=0.8, description="Person-vehicle overlap threshold"),
    triple_threshold: int = Query(3, ge=2, le=5, description="Riders to trigger triple-riding violation"),
    run_alpr: bool = Query(True, description="Run ALPR plate reading on violated vehicles"),
):
    """
    Detect violations in a single uploaded image.
    Returns detections, violations, and annotated image as base64 JPEG.
    """
    raw = await file.read()
    frame = _decode_image(raw)

    # Apply per-request conf threshold without rebuilding the model
    det = get_detector()
    old_conf = det.conf
    det.conf = conf_threshold

    try:
        detections = det.predict(frame)
    finally:
        det.conf = old_conf

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checker = get_checker(overlap=overlap_threshold, triple=triple_threshold)
    violations = checker.check(detections, frame_id=0, timestamp=ts)

    # ALPR on each violation
    if run_alpr and violations:
        try:
            alpr = get_alpr()
            for v in violations:
                alpr_res = _run_alpr_on_violation(frame, v, alpr)
                if alpr_res:
                    v.plate_text = alpr_res.plate_number
                    v.detection_tier = alpr_res.detection_tier
                    v.ocr_confidence_raw = alpr_res.ocr_confidence_raw
                    v.plate_valid = alpr_res.plate_valid
                else:
                    v.plate_text = "UNKNOWN"
                    v.detection_tier = None
                    v.ocr_confidence_raw = None
                    v.plate_valid = False
        except HTTPException:
            pass  # ALPR not installed — leave plate_text as UNKNOWN

    # Save to CSV log
    try:
        from src.utils import save_violation_to_csv
        for v in violations:
            save_violation_to_csv(v, v.plate_text)
    except Exception:
        pass

    # Annotate frame
    try:
        from src.utils import annotate_frame
        annotated = annotate_frame(frame, detections, violations)
    except Exception:
        annotated = frame

    return {
        "detections": [_detection_to_dict(d) for d in detections],
        "violations": [_violation_to_dict(v) for v in violations],
        "annotated_image_b64": _frame_to_b64(annotated),
        "frame_id": 0,
        "timestamp": ts,
    }


# POST /alpr/read
@app.post("/alpr/read", response_model=ALPRResponse, tags=["ALPR"])
async def alpr_read(
    file: UploadFile = File(..., description="Plate crop image"),
):
    """Read an Indian license plate from a cropped plate image."""
    raw = await file.read()
    frame = _decode_image(raw)

    try:
        alpr = get_alpr()
    except HTTPException as e:
        raise e

    try:
        result = alpr.read_plate_from_crop(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    if result is None:
        raise HTTPException(status_code=400, detail="Could not read plate from image.")

    return ALPRResponse(
        raw_text=result.raw_text,
        plate_number=result.plate_number,
        confidence=result.confidence,
        detection_tier=result.detection_tier,
        ocr_confidence_raw=result.ocr_confidence_raw,
        plate_valid=result.plate_valid,
    )


# GET /violations/log
@app.get("/violations/log", response_model=ViolationLogResponse, tags=["Violations"])
async def violations_log(
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
):
    """Return today's violation log, newest first."""
    try:
        from src.utils import load_violations_log
        df = load_violations_log()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log: {e}")

    if df.empty:
        return {"count": 0, "records": []}

    # Newest first, limited
    df = df.iloc[::-1].head(limit)
    records = df.fillna("").to_dict(orient="records")
    return {"count": len(records), "records": records}


# GET /violations/summary
@app.get("/violations/summary", response_model=ViolationSummaryResponse, tags=["Violations"])
async def violations_summary():
    """Return violation counts by type for today."""
    try:
        from src.utils import load_violations_log
        df = load_violations_log()
        
        counts = {
            "total": len(df),
            "NO_HELMET": int((df["violation_type"] == "NO_HELMET").sum()),
            "TRIPLE_RIDING": int((df["violation_type"] == "TRIPLE_RIDING").sum()),
            "SIGNAL_JUMP": int((df["violation_type"] == "SIGNAL_JUMP").sum()),
            "NO_SEATBELT": int((df["violation_type"] == "NO_SEATBELT").sum()),
            "WRONG_WAY": int((df["violation_type"] == "WRONG_WAY").sum()),
        }
        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log: {e}")

# DELETE /violations/log
@app.delete("/violations/log", tags=["Violations"])
async def clear_violations():
    """Delete today's violation log."""
    try:
        from src.utils import clear_violations_log
        clear_violations_log()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear log: {e}")


# WS /ws/detect/video  — Two-stage helmet pipeline (mirrors edge_client.py)
@app.websocket("/ws/detect/video")
async def ws_detect_video(websocket: WebSocket):
    """
    Stream video frames for violation detection using the two-stage pipeline:
      1. COCO YOLOv8s tracks vehicles + persons per frame.
      2. For each motorcycle/bicycle, a specialized helmet YOLO model
         checks cropped head regions for NO_HELMET violations.
      3. Confirmed violations are also POSTed to the Judge Feed (port 8001).

    Protocol:
      1. Client sends raw video bytes as first binary message.
      2. Server processes every 3rd frame, sends JSON per frame:
           {frame_id, progress, violations, annotated_image_b64}
      3. Server sends {"done": true} on completion.
    """
    await websocket.accept()
    tmp_path = None

    try:
        params     = websocket.query_params
        frame_skip = int(params.get("frame_skip", 3))

        # Optional zone/direction config for signal-jump + wrong-way checks.
        # signal_roi: "x1,y1,x2,y2" pixel stop-line box; expected_direction:
        # "up"|"down"|"left"|"right" — the lane's normal direction of travel.
        signal_roi = None
        roi_param  = params.get("signal_roi")
        if roi_param:
            try:
                signal_roi = [int(v) for v in roi_param.split(",")]
                if len(signal_roi) != 4:
                    signal_roi = None
            except ValueError:
                signal_roi = None
        expected_direction = params.get("expected_direction") or None

        # Receive video bytes from browser
        video_bytes = await websocket.receive_bytes()

        # Write to temp file — cv2.VideoCapture needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # Load models (singletons — cached after first call)
        model = _get_coco_model()
        _get_helmet_model()     # pre-warm so first frame isn't slow
        _get_seatbelt_model()

        try:
            alpr = get_alpr()
        except Exception:
            alpr = None

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        cap.release()

        confirmed_ids    : set  = set()   # vehicles already flagged (never re-flag)
        known_violations : dict = {}      # {vehicle_id: violation_type} — persists all video
        track_history    : dict = {}      # {vehicle_id: [(cx,cy), ...]} — feeds wrong-way check
        # Cache the last detected objects so we can annotate skipped frames too
        _last_bikes     : list = []
        _last_persons   : list = []
        _last_motor_vehs: list = []
        frame_id = 0

        cap = cv2.VideoCapture(tmp_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                # Resize wide frames
                h0, w0 = frame.shape[:2]
                if w0 > 1280:
                    scale = 1280 / w0
                    frame = cv2.resize(frame, (1280, int(h0 * scale)))

                # Skip frames for speed — but STILL annotate with cached detections
                # so bounding boxes don't flicker off between processed frames.
                if frame_id % frame_skip != 0:
                    progress = round(min(frame_id / total_frames, 1.0), 4)
                    annotated_skip = _annotate(
                        frame, _last_bikes, _last_persons, _last_motor_vehs, [], known_violations
                    )
                    payload = {
                        "frame_id": frame_id,
                        "progress": progress,
                        "violations": [],
                        "annotated_image_b64": _frame_to_b64(annotated_skip),
                    }
                    await websocket.send_text(json.dumps(payload))
                    continue

                annotated, violations, _last_bikes, _last_persons, _last_motor_vehs = \
                    _two_stage_process_frame(
                        frame, frame_id, model, confirmed_ids, known_violations,
                        signal_roi=signal_roi, expected_direction=expected_direction,
                        track_history=track_history,
                    )

                # Bridge confirmed violations to Judge Feed
                for v in violations:
                    crop_bytes = v.pop("_crop_bytes", b"")
                    if crop_bytes:
                        await _post_to_judge(
                            v["vehicle_id"], v["violation_type"],
                            crop_bytes, frame_id
                        )

                    # ALPR and CSV Logging
                    alpr_res = _run_alpr_on_violation(frame, v, alpr)
                    if alpr_res:
                        v["plate_text"] = alpr_res.plate_number
                        v["detection_tier"] = alpr_res.detection_tier
                        v["ocr_confidence_raw"] = alpr_res.ocr_confidence_raw
                        v["plate_valid"] = alpr_res.plate_valid
                    else:
                        v["plate_text"] = "UNKNOWN"
                        v["detection_tier"] = None
                        v["ocr_confidence_raw"] = None
                        v["plate_valid"] = False

                    try:
                        from src.utils import save_violation_to_csv
                        save_violation_to_csv(v, v["plate_text"])
                    except Exception:
                        pass

                progress = round(min(frame_id / total_frames, 1.0), 4)
                payload  = {
                    "frame_id":           frame_id,
                    "progress":           progress,
                    "violations":         violations,
                    "annotated_image_b64": _frame_to_b64(annotated),
                }
                await websocket.send_text(json.dumps(payload))

        finally:
            cap.release()

        await websocket.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)


def _parse_zone_params(params) -> tuple:
    """Shared query-param parsing for signal_roi + expected_direction."""
    signal_roi = None
    roi_param  = params.get("signal_roi")
    if roi_param:
        try:
            signal_roi = [int(v) for v in roi_param.split(",")]
            if len(signal_roi) != 4:
                signal_roi = None
        except ValueError:
            signal_roi = None
    return signal_roi, (params.get("expected_direction") or None)


@app.websocket("/ws/detect/live")
async def ws_detect_live(websocket: WebSocket):
    """
    Same two-stage pipeline as /ws/detect/video, but the source is a live
    RTSP/IP-camera URL (or 0 for a local webcam) opened server-side instead
    of an uploaded file. Runs until the client disconnects or sends
    {"stop": true}.

    Query params: source (required, e.g. rtsp://user:pass@host/stream),
    frame_skip, signal_roi ("x1,y1,x2,y2"), expected_direction.
    """
    await websocket.accept()
    params     = websocket.query_params
    source     = params.get("source")
    frame_skip = int(params.get("frame_skip", 3))
    signal_roi, expected_direction = _parse_zone_params(params)

    if not source:
        await websocket.send_text(json.dumps({"error": "missing ?source= camera URL"}))
        await websocket.close()
        return

    model = _get_coco_model()
    _get_helmet_model()
    _get_seatbelt_model()
    try:
        alpr = get_alpr()
    except Exception:
        alpr = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        await websocket.send_text(json.dumps({"error": f"could not open camera source: {source}"}))
        await websocket.close()
        return

    confirmed_ids: set = set()
    known_violations: dict = {}
    track_history: dict = {}
    frame_id = 0

    try:
        while True:
            # Non-blocking check for a client-sent stop signal
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if json.loads(msg).get("stop"):
                    break
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.5)   # camera hiccup — retry
                continue
            frame_id += 1

            h0, w0 = frame.shape[:2]
            if w0 > 1280:
                scale = 1280 / w0
                frame = cv2.resize(frame, (1280, int(h0 * scale)))

            if frame_id % frame_skip != 0:
                continue

            annotated, violations, *_ = _two_stage_process_frame(
                frame, frame_id, model, confirmed_ids, known_violations,
                signal_roi=signal_roi, expected_direction=expected_direction,
                track_history=track_history,
            )

            for v in violations:
                crop_bytes = v.pop("_crop_bytes", b"")
                if crop_bytes:
                    await _post_to_judge(v["vehicle_id"], v["violation_type"], crop_bytes, frame_id)
                alpr_res = _run_alpr_on_violation(frame, v, alpr)
                v["plate_text"] = alpr_res.plate_number if alpr_res else "UNKNOWN"
                try:
                    from src.utils import save_violation_to_csv
                    save_violation_to_csv(v, v["plate_text"])
                except Exception:
                    pass

            await websocket.send_text(json.dumps({
                "frame_id": frame_id,
                "live": True,
                "violations": violations,
                "annotated_image_b64": _frame_to_b64(annotated),
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        cap.release()


# POST /dataset/fetch/roboflow
@app.post("/dataset/fetch/roboflow", response_model=DatasetFetchResponse, tags=["Dataset"])
async def dataset_fetch_roboflow(body: DatasetFetchRoboflowRequest):
    """
    Pull a Roboflow dataset version via the roboflow Python package.
    Requires: pip install roboflow
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="roboflow package not installed. Run: pip install roboflow",
        )

    dest = "data/raw/roboflow"
    try:
        rf = Roboflow(api_key=body.api_key)
        project = rf.workspace(body.workspace).project(body.project)
        version = project.version(body.version)
        version.download(body.format, location=dest)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow API error: {e}",
        )

    return {
        "status": "ok",
        "message": f"Downloaded {body.workspace}/{body.project} v{body.version} ({body.format}) to {dest}",
        "path": str(Path(dest).resolve()),
    }


# POST /dataset/fetch/kaggle
@app.post("/dataset/fetch/kaggle", response_model=DatasetFetchResponse, tags=["Dataset"])
async def dataset_fetch_kaggle(
    dataset_slug: str = Query(..., description="Kaggle dataset slug: owner/name"),
):
    """
    Pull a Kaggle dataset via kagglehub.
    Requires kaggle.json in ~/.kaggle/ and pip install kagglehub.
    """
    try:
        import kagglehub
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="kagglehub not installed. Run: pip install kagglehub",
        )

    try:
        path = kagglehub.dataset_download(dataset_slug)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Kaggle API error: {e}",
        )

    return {
        "status": "ok",
        "message": f"Downloaded {dataset_slug} via kagglehub",
        "path": str(path),
    }
