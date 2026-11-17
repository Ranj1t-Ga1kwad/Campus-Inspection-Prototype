"""
ESP32-CAM Road Hazard & Campus Waste Detection System (YOLOv8 / YOLO11)
========================================================================
Optimized for high accuracy, real-time FPS, motion-tolerant tracking,
and robust ESP32-CAM video streaming with zero buffer lag.

Detects:
  - Road Hazards : Potholes and Manholes (via custom trained best.pt)
  - Campus Waste : Plastic Bottles, Chips Wrappers, Cans, Cartons, Cups,
                   and overall Trash (via TACO-trained YOLO model)
"""

import argparse
import sys
import time
import threading
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from ultralytics import YOLO


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "best.pt")
DEFAULT_TRASH_MODEL_PATH = str(Path(__file__).resolve().parent / "weights" / "taco_yolo11s.pt")
DEFAULT_STREAM_URL = "http://192.168.4.1:81/stream"

# Detection confidence thresholds
DEFAULT_CONFIDENCE = 0.50          # Global default confidence threshold
DEFAULT_POTHOLE_CONFIDENCE = 0.50  # Potholes must be distinct road hazards (UNTOUCHED)
DEFAULT_MANHOLE_CONFIDENCE = 0.48  # Manholes have clean geometric shapes (UNTOUCHED)
DEFAULT_TRASH_CONFIDENCE = 0.32    # Responsive threshold for wrappers, bottles, and campus litter
DEFAULT_BOTTLE_CAP_CONF = 0.55     # High threshold to prevent false alarms on light reflections

# Inference settings
DEFAULT_INFERENCE_SIZE = 640       # Standard YOLO resolution for maximum detection accuracy
TRASH_INFERENCE_SIZE = 480         # Lightweight resolution for rapid real-time waste inference
MAX_DETECTIONS = 20

# Road Region & Anti-Hand / Anti-Clutter Geometry Filters (Active by default, toggle with 'F')
DEFAULT_USE_HEURISTIC_FILTERS = True   # Enabled by default to block hands and non-road objects!
MAX_SKIN_COLOR_RATIO = 0.20            # Max allowable skin-tone pixel ratio (rejects hands, fingers, arms)
MAX_SATURATION_MEAN = 85               # Max allowable color saturation (asphalt/concrete has low saturation)
MIN_TEXTURE_VARIATION = 12.0           # Minimum Laplacian gradient variation (rejects flat smooth skin/tables)
ROAD_START_RATIO = 0.18                # Only filter extreme sky/ceiling/horizon (top 18%)
MIN_BOX_AREA_RATIO = 0.0010            # Reject microscopic noise specks
MAX_BOX_AREA_RATIO = 0.55              # Reject giant boxes covering whole frame
MIN_ASPECT_RATIO = 0.28                # Reject extreme thin vertical slits
MAX_ASPECT_RATIO = 3.20                # Reject extreme thin horizontal lines/creases

# Temporal Smoothing & Tracking (can be toggled on/off with 'T' key)
DEFAULT_USE_TEMPORAL = True
TRACKER_MAX_AGE_FRAMES = 8             # Frames to keep a detection visible during brief occlusion
TRACKER_IOU_THRESHOLD = 0.10           # Relaxed IoU for moving cameras
TRACKER_MAX_PIXEL_DISTANCE = 120       # Center distance threshold accounting for camera/vehicle motion


# ============================================================
# ESP32-CAM SENSOR CONTROL & PRESETS
# ============================================================

CAMERA_SIZES = {
    5: ("QVGA", 320, 240),
    6: ("CIF", 400, 296),
    7: ("HVGA", 480, 320),
    8: ("VGA", 640, 480),
    9: ("SVGA", 800, 600),
    10: ("XGA", 1024, 768),
    11: ("HD", 1280, 720),
    12: ("SXGA", 1280, 1024),
    13: ("UXGA", 1600, 1200),
}

# Presets: (framesize_id, jpeg_quality)
CAMERA_PRESETS = {
    "fast": (8, 14),      # 640x480 VGA, smooth 20+ FPS
    "balanced": (8, 11),  # 640x480 VGA, sharp & stable ~18-22 FPS (Recommended)
    "high": (9, 10),      # 800x600 SVGA, high detail ~12-15 FPS
    "max": (10, 10),      # 1024x768 XGA, sharpest but lower frame rate ~8-10 FPS
}

CAMERA_ENHANCE = {
    "lenc": 1,    # Lens correction
    "bpc": 1,     # Bad pixel correction
    "wpc": 1,     # White pixel correction
    "raw_gma": 1, # Gamma curve
    "dcw": 1      # Downsize with decimation
}


def camera_describe(framesize):
    info = CAMERA_SIZES.get(int(framesize))
    return f"{info[0]} ({info[1]}x{info[2]})" if info else f"framesize {framesize}"


def camera_set_param(base_url, var, val):
    try:
        reply = requests.get(f"{base_url}/control", params={"var": var, "val": val}, timeout=3)
        return reply.status_code == 200
    except requests.RequestException:
        return False


def apply_camera_settings(stream_url, framesize, quality, enhance=True):
    """Sends configuration to ESP32-CAM webserver over Wi-Fi."""
    parsed = urlparse(stream_url)
    if not parsed.hostname:
        return False

    base_url = f"http://{parsed.hostname}"
    if parsed.port and parsed.port != 81:
        base_url += f":{parsed.port}"

    print(f"\n[Camera Setup] Connecting to ESP32 control interface at {base_url}...")
    try:
        status_resp = requests.get(f"{base_url}/status", timeout=1.5)
        status_resp.raise_for_status()
        before = status_resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[Camera Setup] Warning: Could not reach {base_url}/status ({type(exc).__name__}).")
        print("[Camera Setup] Stream will run using the camera's current onboard settings.")
        return False

    curr_fs = before.get("framesize", -1)
    curr_q = before.get("quality", -1)
    print(f"[Camera Setup] Current hardware settings: {camera_describe(curr_fs)}, quality={curr_q}")
    print(f"[Camera Setup] Applying target settings : {camera_describe(framesize)}, quality={quality}")

    params_to_set = [("framesize", framesize), ("quality", quality)]
    if enhance:
        params_to_set.extend(CAMERA_ENHANCE.items())

    rejected = []
    for var, val in params_to_set:
        if not camera_set_param(base_url, var, val):
            rejected.append(var)
        time.sleep(0.08)

    if "framesize" in rejected or "quality" in rejected:
        print(f"[Camera Setup] Hardware rejected resolution/quality change. Using current settings.")
        return False

    time.sleep(0.8)

    try:
        after = requests.get(f"{base_url}/status", timeout=3).json()
        new_fs = int(after.get("framesize", -1))
        new_q = int(after.get("quality", -1))
        confirmed = (new_fs == framesize and new_q == quality)
        status_text = "CONFIRMED SUCCESS" if confirmed else "PARTIAL"
        print(f"[Camera Setup] Camera updated: {camera_describe(new_fs)}, quality={new_q} -> [{status_text}]")
        return confirmed
    except Exception:
        print("[Camera Setup] Settings applied; stream restarting.")
        return True


# ============================================================
# ZERO-LAG ESP32-CAM STREAM GRABBER
# ============================================================

class ZeroLagStreamReader:
    """
    High-performance, threaded MJPEG & Video stream grabber.
    Always maintains only the freshest decoded frame in memory,
    completely eliminating network buffer bloat and video delay.
    """

    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_number = 0
        self.running = True
        self.status = "Initializing..."
        self.is_connected = False
        self.is_local_file = False
        self.is_webcam = False

        if str(source).isdigit():
            self.is_webcam = True
            self.source = int(source)
        elif Path(str(source)).exists():
            self.is_local_file = True

        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.running = False

    def snapshot(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            return frame, self.frame_number, self.status, self.is_connected

    def _worker(self):
        while self.running:
            if self.is_webcam or self.is_local_file:
                self._run_opencv_capture()
            else:
                self._run_mjpeg_stream()

            if self.running and not self.is_local_file:
                with self.lock:
                    self.is_connected = False
                    self.status = "Reconnecting in 1s..."
                time.sleep(1.0)
            elif self.is_local_file:
                break

    def _run_opencv_capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            with self.lock:
                self.status = f"Failed to open source {self.source}"
                self.is_connected = False
            time.sleep(1.0)
            return

        with self.lock:
            self.is_connected = True
            self.status = "Camera / Video connected"

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            with self.lock:
                self.latest_frame = frame
                self.frame_number += 1

            if self.is_local_file:
                time.sleep(0.03)

        cap.release()

    def _run_mjpeg_stream(self):
        with self.lock:
            self.status = f"Connecting to {self.source}..."
            self.is_connected = False

        try:
            with requests.get(self.source, stream=True, timeout=(6, 12)) as response:
                if response.status_code != 200:
                    with self.lock:
                        self.status = f"HTTP Error {response.status_code}"
                    time.sleep(1.0)
                    return

                with self.lock:
                    self.status = "ESP32-CAM Connected (Live)"
                    self.is_connected = True

                buffer = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    if not self.running:
                        break
                    if not chunk:
                        continue

                    buffer.extend(chunk)

                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            if len(buffer) > 2:
                                del buffer[:-2]
                            break

                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buffer[:start]
                            if len(buffer) > 500_000:
                                del buffer[:-65_536]
                            break

                        jpeg_bytes = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]

                        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.latest_frame = frame
                                self.frame_number += 1
                                self.is_connected = True

        except requests.RequestException as exc:
            with self.lock:
                self.status = f"Stream Error: {type(exc).__name__}"
                self.is_connected = False
        except Exception as exc:
            with self.lock:
                self.status = f"Reader Error: {str(exc)[:30]}"
                self.is_connected = False


# ============================================================
# MOTION-TOLERANT TRACKER & PERSISTENCE FILTER
# ============================================================

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)

    inter = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def box_center_distance(a, b):
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


class MotionTolerantTracker:
    """
    Maintains persistent track of detected potholes, manholes, and waste items.
    Works even when vehicle/camera is moving forward, preventing
    detection dropouts or rapid blinking.
    """

    def __init__(self, max_age=TRACKER_MAX_AGE_FRAMES, iou_thresh=TRACKER_IOU_THRESHOLD, max_dist=TRACKER_MAX_PIXEL_DISTANCE):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.max_dist = max_dist

    def update(self, detections):
        for trk in self.tracks:
            trk["age"] += 1

        unmatched_dets = list(detections)
        matched_tracks = []

        for trk in self.tracks:
            best_match_idx = None
            best_score = -1.0

            for i, det in enumerate(unmatched_dets):
                if det["label"] != trk["label"]:
                    continue

                iou = box_iou(trk["box"], det["box"])
                dist = box_center_distance(trk["box"], det["box"])

                if iou >= self.iou_thresh or dist <= self.max_dist:
                    score = iou + max(0.0, 1.0 - (dist / self.max_dist))
                    if score > best_score:
                        best_score = score
                        best_match_idx = i

            if best_match_idx is not None:
                matched_det = unmatched_dets.pop(best_match_idx)
                alpha = 0.65
                ox1, oy1, ox2, oy2 = trk["box"]
                nx1, ny1, nx2, ny2 = matched_det["box"]
                smoothed_box = (
                    int(alpha * nx1 + (1 - alpha) * ox1),
                    int(alpha * ny1 + (1 - alpha) * oy1),
                    int(alpha * nx2 + (1 - alpha) * ox2),
                    int(alpha * ny2 + (1 - alpha) * oy2)
                )

                trk["box"] = smoothed_box
                det_conf = matched_det.get("confidence", matched_det.get("conf", 0.0))
                trk["conf"] = max(trk["conf"] * 0.5, det_conf)
                trk["age"] = 0
                trk["hits"] += 1
                matched_tracks.append(trk)

        for det in unmatched_dets:
            det_conf = det.get("confidence", det.get("conf", 0.0))
            new_trk = {
                "id": self.next_id,
                "label": det["label"],
                "box": det["box"],
                "conf": det_conf,
                "confidence": det_conf,
                "age": 0,
                "hits": 1
            }
            self.next_id += 1
            self.tracks.append(new_trk)

        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        active_outputs = []
        for t in self.tracks:
            is_confirmed = (t["hits"] >= 2) or (t["conf"] >= 0.52)
            if is_confirmed and t["age"] <= 3:
                active_outputs.append({
                    "id": t["id"],
                    "label": t["label"],
                    "confidence": t["conf"],
                    "conf": t["conf"],
                    "box": t["box"],
                    "age": t["age"]
                })

        return active_outputs


# ============================================================
# ANTI-HAND, SPECULAR GLARE & PACKAGING VALIDATORS
# ============================================================

def is_specular_glare(patch):
    """
    Identifies specular light reflections, shiny table glare, and overexposed highlights.
    Characteristics:
      - High average brightness (mean V > 190)
      - High percentage (> 22%) of pure blown-out white pixels (V > 220 and S < 35)
    """
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    if mean_v < 185:
        return False
    glare_mask = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 35)
    glare_ratio = float(np.count_nonzero(glare_mask)) / float(patch.shape[0] * patch.shape[1])
    return (glare_ratio > 0.22 and mean_v > 195)


def is_colorful_packaging(patch):
    """
    Checks if a patch has artificial color saturation characteristic of
    wrappers, chips packets, snack bags, cans, or plastic bottles.
    Real road asphalt and potholes are desaturated (S < 25).
    If > 18% of pixels have S > 48 and V > 28, it is a colorful packaging item, NOT a road pothole.
    """
    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    sat_pixels = (s > 48) & (v > 28)
    sat_ratio = float(np.count_nonzero(sat_pixels)) / float(patch.shape[0] * patch.shape[1])
    return sat_ratio > 0.18


def check_skin_and_color(patch):
    """
    Analyzes an image patch to determine if it is human skin (hand, finger, face)
    or non-road packaging/colorful objects.
    """
    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return False, "ok"

    # 1. YCrCb Skin Detection
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8)
    )

    # 2. HSV Skin Detection
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    skin_hsv1 = cv2.inRange(hsv, np.array([0, 20, 35], dtype=np.uint8), np.array([25, 210, 255], dtype=np.uint8))
    skin_hsv2 = cv2.inRange(hsv, np.array([165, 20, 35], dtype=np.uint8), np.array([180, 210, 255], dtype=np.uint8))
    skin_hsv = cv2.bitwise_or(skin_hsv1, skin_hsv2)

    combined_skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    skin_ratio = float(np.count_nonzero(combined_skin)) / float(patch.shape[0] * patch.shape[1])

    if skin_ratio > MAX_SKIN_COLOR_RATIO:
        return True, f"Skin/Hand detected ({skin_ratio*100:.1f}%)"

    # 3. Colorful packaging / wrapper detection (asphalt roads are neutral/grayscale)
    if is_colorful_packaging(patch):
        return True, "Colorful packaging/wrapper detected (non-road)"

    mean_sat = float(np.mean(hsv[:, :, 1]))
    if mean_sat > MAX_SATURATION_MEAN:
        return True, f"High Color Saturation ({mean_sat:.1f})"

    return False, "ok"


# ============================================================
# MULTI-HAZARD & CAMPUS WASTE DETECTOR PIPELINE
# ============================================================

class RoadHazardDetector:
    """
    Encapsulates YOLO inference for road hazards (Potholes, Manholes)
    AND Campus Waste (Bottles, Wrappers, Cans, Cartons, Overall Trash).
    Features smart conflict resolution where trash detections supersede false potholes.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, trash_model_path=DEFAULT_TRASH_MODEL_PATH,
                 imgsz=DEFAULT_INFERENCE_SIZE, conf=DEFAULT_CONFIDENCE,
                 detect_trash=True, trash_conf=DEFAULT_TRASH_CONFIDENCE):
        self.model_path = model_path
        self.trash_model_path = trash_model_path
        self.imgsz = imgsz
        self.conf = conf
        self.pothole_conf = DEFAULT_POTHOLE_CONFIDENCE
        self.manhole_conf = DEFAULT_MANHOLE_CONFIDENCE
        self.trash_conf = trash_conf
        self.use_filters = DEFAULT_USE_HEURISTIC_FILTERS
        self.detect_trash = detect_trash

        # 1. Road Hazard Model (Potholes & Manholes)
        print(f"\n[Model Setup] Loading Pothole & Manhole model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"[Model Setup] Hazard model classes: {self.class_names}")
        self.class_map = self._build_class_map(self.class_names)
        print(f"[Model Setup] Normalized hazard class mapping: {self.class_map}")

        # 2. Campus Trash & Waste Model (Bottles, Wrappers, Litter)
        self.trash_model = None
        self.trash_class_map = {}
        if self.detect_trash:
            self._init_trash_model(trash_model_path)

    def _init_trash_model(self, path):
        """Locates or downloads the campus trash/waste model."""
        target = Path(path)
        if not target.exists():
            script_weights = Path(__file__).resolve().parent / "weights" / "taco_yolo11s.pt"
            alt_weights = Path(__file__).resolve().parent / "weights" / "waste_yolov8n.pt"
            if script_weights.exists():
                target = script_weights
            elif alt_weights.exists():
                target = alt_weights
            else:
                print(f"[Model Setup] Trash model not found at {path}. Attempting download...")
                try:
                    import urllib.request
                    script_weights.parent.mkdir(parents=True, exist_ok=True)
                    url = "https://huggingface.co/fabiocigaina/TACO-yolo11s/resolve/main/best_model.pt"
                    print("[Model Setup] Downloading TACO campus waste detection weights...")
                    urllib.request.urlretrieve(url, str(script_weights))
                    target = script_weights
                except Exception as e:
                    print(f"[Model Setup] Notice: Could not download trash model ({e}). Continuing with pothole detection only.")
                    target = None

        if target and target.exists():
            try:
                print(f"[Model Setup] Loading Campus Waste model: {target}")
                self.trash_model = YOLO(str(target))
                self.trash_class_map = self._build_trash_class_map(self.trash_model.names)
                print(f"[Model Setup] Campus Waste classes active: {self.trash_class_map}\n")
            except Exception as e:
                print(f"[Model Setup] Warning: Failed to load trash model ({e}). Continuing with pothole detection only.\n")
                self.trash_model = None

    def _build_class_map(self, names):
        """Intelligently map hazard class IDs to canonical 'POTHOLE' or 'MANHOLE'."""
        mapping = {}
        for class_id, raw_name in names.items():
            name_str = str(raw_name).strip().lower().replace("_", " ").replace("-", " ")
            tokens = set(name_str.split())

            if any(term in name_str for term in ["manhole", "cover", "lid", "drain", "sewer"]):
                mapping[int(class_id)] = "MANHOLE"
            elif any(term in name_str for term in ["pothole", "d40", "crater"]) or "hole" in tokens:
                mapping[int(class_id)] = "POTHOLE"
            else:
                if len(names) == 2:
                    mapping[int(class_id)] = "POTHOLE" if int(class_id) == 0 else "MANHOLE"
                elif len(names) == 1:
                    mapping[int(class_id)] = "POTHOLE"
                else:
                    mapping[int(class_id)] = name_str.upper()
        return mapping

    def _build_trash_class_map(self, names):
        """
        Intelligently maps trash/waste model class IDs to canonical display labels:
        - WRAPPER (chips packets, snack wrappers, chocolate wrappers, plastic pouches, cartons/packaging)
        - BOTTLE (plastic bottles, drink bottles)
        - BOTTLE CAP (bottle caps, guarded with high threshold & glare filter)
        - CAN (beverage cans, aluminum tins)
        - CUP (disposable cups, paper cups)
        - PLASTIC WASTE (miscellaneous plastic fragments)
        - TRASH (general waste, litter)
        """
        mapping = {}
        for class_id, raw_name in names.items():
            name_str = str(raw_name).strip().lower().replace("_", " ").replace("-", " ")
            if any(w in name_str for w in ["wrapper", "bag", "film", "packet", "pouch", "chips", "crisp", "carton", "box"]):
                mapping[int(class_id)] = "WRAPPER"
            elif any(w in name_str for w in ["cap", "lid"]):
                mapping[int(class_id)] = "BOTTLE CAP"
            elif any(w in name_str for w in ["bottle"]):
                mapping[int(class_id)] = "BOTTLE"
            elif any(w in name_str for w in ["can", "tin"]):
                mapping[int(class_id)] = "CAN"
            elif any(w in name_str for w in ["cup", "mug"]):
                mapping[int(class_id)] = "CUP"
            elif any(w in name_str for w in ["cigarette", "butt"]):
                mapping[int(class_id)] = "CIGARETTE"
            elif any(w in name_str for w in ["peel", "banana", "orange", "food", "bio"]):
                mapping[int(class_id)] = "ORGANIC WASTE"
            elif any(w in name_str for w in ["plastic"]):
                mapping[int(class_id)] = "PLASTIC WASTE"
            elif any(w in name_str for w in ["paper"]):
                mapping[int(class_id)] = "PAPER"
            elif any(w in name_str for w in ["metal"]):
                mapping[int(class_id)] = "METAL"
            elif any(w in name_str for w in ["glass"]):
                mapping[int(class_id)] = "GLASS"
            elif any(w in name_str for w in ["styrofoam", "foam"]):
                mapping[int(class_id)] = "STYROFOAM"
            else:
                mapping[int(class_id)] = "TRASH"
        return mapping

    def predict(self, frame, debug=False):
        """
        Runs YOLO detection on frame:
        1. Road Hazards (potholes, manholes) with full existing heuristic filters
        2. Campus Waste (bottles, wrappers, trash) with customized trash & glare filters
        3. Conflict Resolution: Trash detections override false pothole classifications!
        """
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        raw_detections = []
        pothole_candidates = []
        trash_candidates = []

        # ========================================================
        # 1. POTHOLE & MANHOLE DETECTION
        # ========================================================
        min_pothole_conf = min(self.conf, self.pothole_conf, self.manhole_conf)
        pothole_results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=min_pothole_conf,
            max_det=MAX_DETECTIONS,
            verbose=False
        )[0]

        if pothole_results.boxes is not None and len(pothole_results.boxes) > 0:
            for box in pothole_results.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = self.class_map.get(class_id, "POTHOLE")

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                raw_item = {
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                    "class_id": class_id
                }
                raw_detections.append(raw_item)

                req_conf = self.pothole_conf if label == "POTHOLE" else self.manhole_conf
                effective_conf = max(req_conf, self.conf)

                if confidence < effective_conf:
                    continue

                # Road heuristic filtering (Active for potholes and manholes)
                if self.use_filters:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area_ratio = (box_w * box_h) / frame_area

                    # 1. Filter microscopic specks or full frame boxes
                    if not (MIN_BOX_AREA_RATIO <= box_area_ratio <= MAX_BOX_AREA_RATIO):
                        if debug:
                            print(f"[Filter Rejection] Box ({x1},{y1}) area ratio {box_area_ratio:.4f} invalid")
                        continue

                    # 2. Filter detections in sky / ceiling zone
                    if (y1 + y2) / 2.0 < (h * ROAD_START_RATIO):
                        if debug:
                            print(f"[Filter Rejection] Box ({x1},{y1}) in upper sky/ceiling zone")
                        continue

                    # 3. Class-specific aspect ratio filter
                    aspect_ratio = box_w / float(max(1, box_h))
                    if label == "MANHOLE":
                        if not (0.45 <= aspect_ratio <= 2.20):
                            if debug:
                                print(f"[Filter Rejection] Manhole aspect ratio {aspect_ratio:.2f} invalid")
                            continue
                    else:
                        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                            if debug:
                                print(f"[Filter Rejection] Pothole aspect ratio {aspect_ratio:.2f} invalid")
                            continue

                    # 4. Anti-Hand, Anti-Skin & Non-Road Colorful Packaging Filter
                    patch = frame[y1:y2, x1:x2]
                    is_rejected, reason = check_skin_and_color(patch)
                    if is_rejected:
                        if debug:
                            print(f"[Filter Rejection] Box ({x1},{y1}) rejected: {reason}")
                        continue

                    # 5. Texture Complexity Filter
                    if patch.size > 0 and confidence < 0.65:
                        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                        lap_var = float(cv2.Laplacian(gray_patch, cv2.CV_64F).var())
                        if lap_var < MIN_TEXTURE_VARIATION:
                            if debug:
                                print(f"[Filter Rejection] Box ({x1},{y1}) rejected as smooth non-road surface (var: {lap_var:.1f})")
                            continue

                pothole_candidates.append({
                    "label": label,
                    "confidence": confidence,
                    "conf": confidence,
                    "box": (x1, y1, x2, y2)
                })

        # ========================================================
        # 2. CAMPUS TRASH & WASTE DETECTION
        # ========================================================
        if self.detect_trash and self.trash_model is not None:
            effective_trash_conf = min(self.trash_conf, max(0.25, self.conf - 0.15))
            trash_results = self.trash_model.predict(
                source=frame,
                imgsz=TRASH_INFERENCE_SIZE,
                conf=effective_trash_conf,
                max_det=MAX_DETECTIONS,
                verbose=False
            )[0]

            if trash_results.boxes is not None and len(trash_results.boxes) > 0:
                for box in trash_results.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    label = self.trash_class_map.get(class_id, "TRASH")

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    raw_item = {
                        "label": label,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                        "class_id": class_id
                    }
                    raw_detections.append(raw_item)

                    # Category-specific confidence threshold
                    # Bottle caps require higher confidence (>= 0.55) to prevent light reflection false alarms
                    req_trash_conf = DEFAULT_BOTTLE_CAP_CONF if label == "BOTTLE CAP" else effective_trash_conf
                    if confidence < req_trash_conf:
                        continue

                    patch = frame[y1:y2, x1:x2]

                    # Specular light reflection / glare rejection filter
                    if is_specular_glare(patch):
                        if debug:
                            print(f"[Filter Rejection] Trash Box ({x1},{y1}) rejected: Specular light reflection/glare")
                        continue

                    # Trash-specific geometric validation
                    if self.use_filters:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_area_ratio = (box_w * box_h) / frame_area

                        if not (0.0004 <= box_area_ratio <= 0.65):
                            continue

                        if (y1 + y2) / 2.0 < (h * 0.15):
                            continue

                        # Anti-bare-hand check (only reject if dominant skin tone > 60%)
                        if patch.size > 0 and patch.shape[0] >= 10 and patch.shape[1] >= 10:
                            ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
                            skin_mask = cv2.inRange(
                                ycrcb,
                                np.array([0, 133, 77], dtype=np.uint8),
                                np.array([255, 173, 127], dtype=np.uint8)
                            )
                            skin_ratio = float(np.count_nonzero(skin_mask)) / float(patch.shape[0] * patch.shape[1])
                            if skin_ratio > 0.60 and confidence < 0.65:
                                if debug:
                                    print(f"[Filter Rejection] Trash Box ({x1},{y1}) rejected as skin/hand ({skin_ratio*100:.1f}%)")
                                continue

                    trash_candidates.append({
                        "label": label,
                        "confidence": confidence,
                        "conf": confidence,
                        "box": (x1, y1, x2, y2)
                    })

        # ========================================================
        # 3. SMART CONFLICT RESOLUTION: TRASH SUPERSEDES POTHOLES
        # ========================================================
        # If both models detect an object, the specialized trash model
        # takes priority over the generic dark-patch pothole detector.
        validated_potholes = []
        for p in pothole_candidates:
            is_overridden_by_trash = False
            for t in trash_candidates:
                iou = box_iou(p["box"], t["box"])
                center_dist = box_center_distance(p["box"], t["box"])
                if iou > 0.25 or center_dist < 80:
                    is_overridden_by_trash = True
                    if debug:
                        print(f"[Conflict Resolution] Pothole at {p['box']} overridden by {t['label']} ({t['confidence']:.2f})")
                    break
            if not is_overridden_by_trash:
                validated_potholes.append(p)

        detections = validated_potholes + trash_candidates
        return detections, raw_detections


# ============================================================
# HIGH-AESTHETIC HEADS-UP DISPLAY (HUD)
# ============================================================

CLASS_COLORS = {
    "POTHOLE": (0, 140, 255),       # Neon Orange
    "MANHOLE": (0, 220, 110),       # Vivid Emerald Green
    "BOTTLE": (255, 200, 0),        # Electric Cyan / Sky Blue
    "BOTTLE CAP": (230, 180, 50),   # Light Cyan
    "WRAPPER": (180, 50, 255),      # Vivid Neon Magenta / Hot Pink (Chips packets, wrappers)
    "CAN": (0, 215, 255),           # Bright Amber / Gold
    "CARTON": (0, 165, 255),        # Warm Gold
    "CUP": (235, 175, 40),          # Sky Blue
    "CIGARETTE": (140, 140, 240),   # Soft Coral
    "PLASTIC WASTE": (50, 230, 100),# Bright Spring Green
    "ORGANIC WASTE": (80, 200, 120),# Foliage Green
    "TRASH": (30, 220, 255),        # Bright Yellow-Lime
}


def get_label_color(label):
    return CLASS_COLORS.get(label, (30, 220, 255))


def draw_hud(frame, detections, status, fps, inference_ms, conf_thresh, use_temporal, use_filter,
             debug_mode, raw_count=0, detect_trash=True):
    """Renders modern glassmorphism UI overlay with detections and metrics."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Hazard counts
    potholes_count = sum(1 for d in detections if d["label"] == "POTHOLE")
    manholes_count = sum(1 for d in detections if d["label"] == "MANHOLE")

    # Campus waste counts
    trash_items = [d for d in detections if d["label"] not in ("POTHOLE", "MANHOLE")]
    trash_count = len(trash_items)
    bottles_count = sum(1 for d in trash_items if d["label"] == "BOTTLE")
    wrappers_count = sum(1 for d in trash_items if d["label"] == "WRAPPER")

    COLOR_PANEL_BG = (22, 22, 26)       # Deep charcoal
    COLOR_TEXT_MAIN = (255, 255, 255)   # White
    COLOR_TEXT_MUTED = (180, 180, 190)  # Light gray

    # 1. Draw Bounding Boxes with Tactical Badges
    has_close_hazard = False
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]
        color = get_label_color(label)

        if (y1 + y2) / 2.0 > h * 0.65 and label == "POTHOLE":
            has_close_hazard = True

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        corner_len = max(8, min(18, (x2 - x1) // 4, (y2 - y1) // 4))
        cv2.line(out, (x1, y1), (x1 + corner_len, y1), color, 4, cv2.LINE_AA)
        cv2.line(out, (x1, y1), (x1, y1 + corner_len), color, 4, cv2.LINE_AA)
        cv2.line(out, (x2, y1), (x2 - corner_len, y1), color, 4, cv2.LINE_AA)
        cv2.line(out, (x2, y1), (x2, y1 + corner_len), color, 4, cv2.LINE_AA)
        cv2.line(out, (x1, y2), (x1 + corner_len, y2), color, 4, cv2.LINE_AA)
        cv2.line(out, (x1, y2), (x1, y2 - corner_len), color, 4, cv2.LINE_AA)
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, 4, cv2.LINE_AA)
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, 4, cv2.LINE_AA)

        label_text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)

        badge_y1 = max(0, y1 - th - 10)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(w, x1 + tw + 12)

        cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            out,
            label_text,
            (badge_x1 + 6, badge_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # 2. Imminent Pothole Warning Banner
    if has_close_hazard:
        banner_text = ">> CAUTION: POTHOLE AHEAD <<"
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        bx = (w - bw) // 2
        by = 85
        cv2.rectangle(out, (bx - 14, by - bh - 8), (bx + bw + 14, by + 10), (0, 0, 200), -1)
        cv2.rectangle(out, (bx - 14, by - bh - 8), (bx + bw + 14, by + 10), (0, 220, 255), 2)
        cv2.putText(out, banner_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Top Status Dashboard (Glassmorphism Dark Panel)
    panel_w = min(640, w - 20)
    panel_h = 46
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 8), (10 + panel_w, 8 + panel_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.rectangle(out, (10, 8), (10 + panel_w, 8 + panel_h), (60, 60, 70), 1)

    # Top line: Potholes, Manholes, and Campus Trash counts
    cv2.circle(out, (24, 31), 6, CLASS_COLORS["POTHOLE"], -1)
    cv2.putText(out, f"Potholes: {potholes_count}", (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    cv2.circle(out, (140, 31), 6, CLASS_COLORS["MANHOLE"], -1)
    cv2.putText(out, f"Manholes: {manholes_count}", (151, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    if detect_trash:
        cv2.circle(out, (258, 31), 6, CLASS_COLORS["WRAPPER"], -1)
        trash_label_str = f"Trash: {trash_count}"
        if trash_count > 0:
            sub_details = []
            if wrappers_count > 0:
                sub_details.append(f"{wrappers_count} Wrp")
            if bottles_count > 0:
                sub_details.append(f"{bottles_count} Btl")
            rem = trash_count - bottles_count - wrappers_count
            if rem > 0:
                sub_details.append(f"{rem} Wst")
            if sub_details:
                trash_label_str += f" ({', '.join(sub_details)})"
        cv2.putText(out, trash_label_str, (269, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # Top right metrics: Right-aligned cleanly to prevent collision
    fps_str = f"FPS: {fps:.1f} ({inference_ms:.0f}ms)"
    (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    fps_x = 10 + panel_w - fps_w - 14
    cv2.putText(out, fps_str, (fps_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)

    if debug_mode:
        dbg_text = f"DBG: {raw_count} raw"
        cv2.putText(out, dbg_text, (fps_x - 90, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    # 4. Bottom Information Bar
    bot_y = h - 12
    status_disp = f"Status: {status}"
    cv2.putText(out, status_disp[:55], (12, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    help_disp = "[+/-] Conf  [C] Trash  [T] Track  [F] Filter  [D] Debug  [S] Save  [Q] Quit"
    (hw, _), _ = cv2.getTextSize(help_disp, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    if w > hw + 240:
        cv2.putText(out, help_disp, (w - hw - 12, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 180, 190), 1, cv2.LINE_AA)

    return out


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM Real-Time Road Hazard & Campus Waste Detection System")
    parser.add_argument("--source", default=DEFAULT_STREAM_URL,
                        help="Video source: ESP32 stream URL, webcam index (0, 1), or video file path")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to YOLO best.pt pothole & manhole model")
    parser.add_argument("--trash-model", default=DEFAULT_TRASH_MODEL_PATH,
                        help="Path to YOLO campus trash & waste model")
    parser.add_argument("--no-trash", action="store_true",
                        help="Disable campus trash/waste detection (potholes only)")
    parser.add_argument("--trash-conf", type=float, default=DEFAULT_TRASH_CONFIDENCE,
                        help="Confidence threshold for trash/waste detection (default: 0.32)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                        help="Initial confidence threshold (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_INFERENCE_SIZE,
                        help="Inference image resolution (default: 640)")
    parser.add_argument("--preset", choices=sorted(CAMERA_PRESETS), default="balanced",
                        help="ESP32-CAM sensor preset (balanced=640x480 quality 11, fast=640x480 quality 14, high=800x600)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip sending HTTP setup commands to ESP32-CAM")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal smoothing tracker")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable anti-hand, color, and road geometry heuristic filters")
    parser.add_argument("--save-dir", default="detections_saved",
                        help="Directory to save screenshots")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        script_dir = Path(__file__).resolve().parent
        if (script_dir / "best.pt").exists():
            model_path = script_dir / "best.pt"
        elif Path("best.pt").exists():
            model_path = Path("best.pt").resolve()
        else:
            print(f"\n[ERROR] Model file not found at: {model_path.resolve()}")
            print("Please ensure 'best.pt' is in this directory or provide --model path/to/best.pt")
            if sys.platform == "win32" and sys.stdin.isatty():
                try:
                    input("\nPress Enter to exit...")
                except Exception:
                    pass
            sys.exit(1)

    detector = RoadHazardDetector(
        model_path=str(model_path),
        trash_model_path=str(args.trash_model),
        imgsz=args.imgsz,
        conf=args.conf,
        detect_trash=not args.no_trash,
        trash_conf=args.trash_conf
    )
    detector.use_filters = not args.no_filter

    if args.source == DEFAULT_STREAM_URL:
        print("[Source Setup] Checking ESP32-CAM stream connectivity at 192.168.4.1 (1.0s timeout)...")
        esp32_online = False
        try:
            r = requests.get("http://192.168.4.1/status", timeout=1.0)
            if r.status_code == 200:
                esp32_online = True
        except Exception:
            esp32_online = False

        if not esp32_online:
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                test_cap.release()
                print("[Source Setup] ESP32-CAM not detected on Wi-Fi.")
                print("[Source Setup] -> Auto-switching to Laptop Webcam (source: 0).")
                print("                 (Press 'E' during playback or run with --source http://192.168.4.1:81/stream for ESP32)\n")
                args.source = "0"
            else:
                print("[Source Setup] ESP32-CAM not responding and Webcam 0 not detected.")
                print(f"[Source Setup] -> Continuing with stream URL: {args.source}\n")

    is_http_stream = str(args.source).startswith("http://") or str(args.source).startswith("https://")
    if is_http_stream and not args.skip_setup:
        preset_fs, preset_q = CAMERA_PRESETS[args.preset]
        apply_camera_settings(args.source, framesize=preset_fs, quality=preset_q)

    print(f"[Stream Reader] Starting video capture from: {args.source}")
    reader = ZeroLagStreamReader(args.source)
    reader.start()

    use_temporal = not args.no_temporal
    tracker = MotionTolerantTracker()

    window_name = "AI Road Hazard & Campus Waste Detector (Potholes & Trash)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    fps = 0.0
    fps_counter = 0
    fps_start_time = time.perf_counter()
    inference_ms = 0.0

    last_sequence = -1
    last_output_frame = None
    debug_mode = False
    is_paused = False

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("  AI ROAD HAZARD & CAMPUS WASTE DETECTION SYSTEM ONLINE")
    print("=" * 68)
    print("  Interactive Controls:")
    print("    [+] or [=]  : Increase confidence threshold (+0.05)")
    print("    [-] or [_]  : Decrease confidence threshold (-0.05)")
    print("    [C]         : Toggle Campus Waste & Trash detection on/off")
    print("    [T]         : Toggle temporal tracker smoothing on/off")
    print("    [F]         : Toggle road heuristic filter on/off")
    print("    [D]         : Toggle debug diagnostic mode")
    print("    [S]         : Save current frame screenshot")
    print("    [SPACE]     : Pause / Resume playback")
    print("    [Q] / [ESC] : Quit application")
    print("=" * 68 + "\n")

    try:
        while True:
            if not is_paused:
                frame, seq, status, connected = reader.snapshot()

                if frame is not None and seq != last_sequence:
                    last_sequence = seq

                    t0 = time.perf_counter()
                    detections, raw_detections = detector.predict(frame, debug=debug_mode)
                    inference_ms = (time.perf_counter() - t0) * 1000.0

                    if use_temporal:
                        tracked_dets = tracker.update(detections)
                        display_dets = tracked_dets
                    else:
                        display_dets = detections

                    last_output_frame = draw_hud(
                        frame=frame,
                        detections=display_dets,
                        status=status,
                        fps=fps,
                        inference_ms=inference_ms,
                        conf_thresh=detector.conf,
                        use_temporal=use_temporal,
                        use_filter=detector.use_filters,
                        debug_mode=debug_mode,
                        raw_count=len(raw_detections),
                        detect_trash=detector.detect_trash
                    )

                    fps_counter += 1
                    elapsed = time.perf_counter() - fps_start_time
                    if elapsed >= 0.8:
                        fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_start_time = time.perf_counter()

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[Window] GUI window closed by user.")
                    break
            except Exception:
                pass

            if last_output_frame is not None:
                cv2.imshow(window_name, last_output_frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                placeholder = np.zeros((480, 720, 3), dtype=np.uint8)
                cv2.putText(placeholder, "AI ROAD HAZARD & CAMPUS WASTE DETECTOR", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(placeholder, f"Active Source: {args.source}", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(placeholder, f"Status: {status}", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1, cv2.LINE_AA)

                cv2.putText(placeholder, "Troubleshooting & Quick Camera Selection:", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(placeholder, "  [W] or [0] : Switch to Laptop Webcam immediately", (30, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 240, 120), 1, cv2.LINE_AA)
                cv2.putText(placeholder, "  [E]        : Switch to ESP32-CAM Wi-Fi stream (192.168.4.1)", (30, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(placeholder, "1. If using ESP32, connect laptop Wi-Fi to ESP32-CAM AP.", (30, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(placeholder, "2. Check ESP32 power and that camera module is connected.", (30, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(placeholder, "Press [Q] or [ESC] to quit.", (30, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 150), 1, cv2.LINE_AA)
                cv2.imshow(window_name, placeholder)
                key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("\nExiting detection...")
                break

            elif key in (ord('c'), ord('C')):
                detector.detect_trash = not detector.detect_trash
                state_str = "ENABLED" if detector.detect_trash else "DISABLED"
                print(f"[Toggle] Campus Waste & Trash Detection: {state_str}")

            elif key in (ord('w'), ord('W'), ord('0')):
                print("\n[Input] Switching camera source to local Webcam (0)...")
                reader.stop()
                args.source = "0"
                reader = ZeroLagStreamReader(0)
                reader.start()
                last_output_frame = None

            elif key in (ord('e'), ord('E')):
                print("\n[Input] Switching camera source to ESP32-CAM stream...")
                reader.stop()
                args.source = DEFAULT_STREAM_URL
                reader = ZeroLagStreamReader(DEFAULT_STREAM_URL)
                reader.start()
                last_output_frame = None

            elif key in (ord('+'), ord('='), ord(']')):
                detector.conf = min(0.95, round(detector.conf + 0.05, 2))
                print(f"[Adjust] Confidence threshold increased to: {detector.conf:.2f}")

            elif key in (ord('-'), ord('_'), ord('[')):
                detector.conf = max(0.10, round(detector.conf - 0.05, 2))
                print(f"[Adjust] Confidence threshold decreased to: {detector.conf:.2f}")

            elif key in (ord('t'), ord('T')):
                use_temporal = not use_temporal
                state_str = "ENABLED" if use_temporal else "DISABLED"
                print(f"[Toggle] Motion-Tolerant Tracker: {state_str}")

            elif key in (ord('f'), ord('F')):
                detector.use_filters = not detector.use_filters
                state_str = "ENABLED" if detector.use_filters else "DISABLED"
                print(f"[Toggle] Heuristic Geometry Filters: {state_str}")

            elif key in (ord('d'), ord('D')):
                debug_mode = not debug_mode
                state_str = "ENABLED" if debug_mode else "DISABLED"
                print(f"[Toggle] Debug Diagnostic Mode: {state_str}")

            elif key == ord(' '):
                is_paused = not is_paused
                print(f"[Action] {'PAUSED' if is_paused else 'RESUMED'}")

            elif key in (ord('s'), ord('S')) and last_output_frame is not None:
                timestamp = int(time.time())
                save_path = save_dir / f"hazard_detection_{timestamp}.jpg"
                if cv2.imwrite(str(save_path), last_output_frame):
                    print(f"[Screenshot] Saved: {save_path.resolve()}")

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("[Done] Camera stream stopped and windows closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
    except Exception as exc:
        import traceback
        print("\n" + "=" * 65)
        print("  [ERROR] An unexpected error occurred:")
        print("=" * 65)
        traceback.print_exc()
        print("=" * 65)
    finally:
        if sys.platform == "win32" and sys.stdin.isatty():
            try:
                input("\nPress Enter to exit...")
            except (EOFError, KeyboardInterrupt):
                pass
