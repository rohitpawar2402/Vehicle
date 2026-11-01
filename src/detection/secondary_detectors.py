from collections import defaultdict
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.heuristics.rider_filter import RiderHead


class SecondaryViolationDetector:
    """Detects Helmets and Seatbelts with visual bounding-box extraction."""

    def __init__(
        self,
        helmet_weights: str = "models/yolov8n_helmet.pt",
        seatbelt_weights: str = "models/yolov8s_seatbelt.pt",
        device: str = "cuda:0",
        half: bool = True,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = half and self.device.startswith("cuda")
        self.helmet_model = self._load_model(helmet_weights)
        self.seatbelt_model = self._load_model(seatbelt_weights)

        # Temporal state hysteresis to prevent helmet prediction flickering
        self.helmet_history: Dict[int, List[float]] = defaultdict(list)
        self.locked_status: Dict[int, str] = {}

    def _apply_hysteresis(self, track_id: Optional[int], instant_is_helmet: bool) -> str:
        """Eliminates helmet prediction flicker using temporal state hysteresis and locking."""
        if track_id is None:
            return "HELMET" if instant_is_helmet else "NO_HELMET"
        if track_id in self.locked_status:
            return self.locked_status[track_id]

        history = self.helmet_history[track_id]
        history.append(1.0 if instant_is_helmet else 0.0)
        if len(history) > 10:
            history.pop(0)

        helmet_ratio = sum(history) / len(history)
        if len(history) >= 3 and helmet_ratio >= 0.50:
            self.locked_status[track_id] = "HELMET"
            return "HELMET"
        return "HELMET" if helmet_ratio >= 0.50 else "NO_HELMET"

    def _load_model(self, path: str) -> Optional[YOLO]:
        if not os.path.exists(path):
            return None
        try:
            model = YOLO(path)
            if self.device.startswith("cuda"):
                model.to(self.device)
            return model
        except Exception:
            return None

    def inspect_motorcycle_helmets(
        self,
        frame: np.ndarray,
        bike_bbox: List[float],
        person_boxes: Optional[List[List[float]]] = None,
        conf_thresh: float = 0.35,
        track_id: Optional[int] = None,
    ) -> List[RiderHead]:
        """Extracts rider head coordinates and evaluates helmet compliance."""
        x1, y1, x2, y2 = [int(v) for v in bike_bbox]
        h_frame, w_frame = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1

        # 1. If custom weights are loaded, run inference
        if self.helmet_model is not None:
            rider_crop = frame[max(0, y1) : min(h_frame, y1 + int(bh * 0.70)), max(0, x1) : min(w_frame, x2)]
            if rider_crop.size > 0:
                results = self.helmet_model(
                    rider_crop, conf=conf_thresh, verbose=False, device=self.device
                )[0]
                heads = []
                if results.boxes is not None and len(results.boxes) > 0:
                    for box, cls_id, conf in zip(
                        results.boxes.xyxy.cpu().numpy(),
                        results.boxes.cls.cpu().numpy(),
                        results.boxes.conf.cpu().numpy(),
                    ):
                        global_box = [float(box[0] + x1), float(box[1] + y1), float(box[2] + x1), float(box[3] + y1)]
                        cname = self.helmet_model.names.get(int(cls_id), "").lower()
                        instant_is_helmet = ("with" in cname and "without" not in cname) or (cname == "helmet") or (int(cls_id) == 0 and "helmet" in cname)
                        label = self._apply_hysteresis(track_id, instant_is_helmet)
                        heads.append(RiderHead(bbox=global_box, label=label, confidence=float(conf)))
                if heads:
                    return heads

        # 2. Check if a detected person box sits on this bike
        if person_boxes:
            for pbox in person_boxes:
                px1, py1, px2, py2 = [int(v) for v in pbox]
                inter_x1, inter_y1 = max(x1, px1), max(y1, py1)
                inter_x2, inter_y2 = min(x2, px2), min(y2, py2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    # Head is upper 28% of the person box
                    p_h = py2 - py1
                    head_h = max(18, int(p_h * 0.28))
                    hx1, hy1 = max(0, px1), max(0, py1)
                    hx2, hy2 = min(w_frame, px2), min(h_frame, py1 + head_h)
                    head_crop = frame[hy1:hy2, hx1:hx2]
                    if head_crop.size > 0:
                        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        std_dev = float(np.std(gray_head))
                        instant_is_helmet = std_dev > 28.0
                        label = self._apply_hysteresis(track_id, instant_is_helmet)
                        conf = 0.88 if label == "HELMET" else 0.74
                        return [RiderHead(bbox=[hx1, hy1, hx2, hy2], label=label, confidence=conf)]

        # 3. Adaptive Heuristic Head Extraction (Rider sits in the top 28% of the bike bounding box)
        head_w = int(bw * 0.45)
        head_h = int(bh * 0.28)
        hx1 = max(0, x1 + int((bw - head_w) / 2))
        hx2 = min(w_frame, hx1 + head_w)
        hy1 = max(0, y1 - int(head_h * 0.15))
        hy2 = min(h_frame, hy1 + head_h)

        head_crop = frame[hy1:hy2, hx1:hx2]
        if head_crop.size == 0:
            return []

        # Analyze color variance / brightness of head crop (helmets have distinct specular reflection/texture)
        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray_head))
        instant_is_helmet = std_dev > 30.0
        label = self._apply_hysteresis(track_id, instant_is_helmet)
        conf = 0.84 if label == "HELMET" else 0.76

        return [RiderHead(bbox=[hx1, hy1, hx2, hy2], label=label, confidence=conf)]

    def inspect_car_seatbelt(self, cabin_crop: np.ndarray, conf_thresh: float = 0.60) -> Tuple[bool, float]:
        """Classifies seatbelt compliance on the CLAHE-enhanced windshield crop."""
        if cabin_crop.size == 0:
            return False, 0.0

        if self.seatbelt_model is not None:
            results = self.seatbelt_model(
                cabin_crop, conf=conf_thresh, verbose=False, device=self.device, imgsz=384
            )[0]
            is_violation = False
            highest_conf = 0.0
            if results.boxes is not None and len(results.boxes) > 0:
                for cls_id, conf in zip(results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
                    if int(cls_id) == 0 and conf > highest_conf:
                        is_violation = True
                        highest_conf = float(conf)
                    elif int(cls_id) == 1 and conf > highest_conf:
                        is_violation = False
                        highest_conf = float(conf)
            return is_violation, highest_conf

        # Fallback: diagonal edge variance heuristic on CLAHE crop
        gray = cv2.cvtColor(cabin_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        diagonal_density = float(np.sum(edges) / edges.size)
        is_violation = diagonal_density < 16.0
        return is_violation, 0.78
