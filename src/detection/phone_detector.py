from enum import Enum
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class DriverSide(str, Enum):
    RIGHT = "RIGHT"  # India, UK, Australia, Japan
    LEFT = "LEFT"    # US, EU, Middle East


class MobilePhoneDetector:
    """Detects driver mobile phone usage within the CLAHE-enhanced cabin crop."""

    def __init__(
        self,
        weights_path: str = "models/yolov8s_phone.pt",
        driver_side: DriverSide = DriverSide.RIGHT,
        device: str = "cuda:0",
        half: bool = True,
    ):
        self.driver_side = driver_side
        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = half and self.device.startswith("cuda")
        self.model = self._load_model(weights_path)

    def _load_model(self, path: str) -> Optional[YOLO]:
        if not os.path.exists(path):
            print(f"[WARN] Phone detector weights '{path}' not found on disk. Running in fallback mode.")
            return None
        try:
            model = YOLO(path)
            if self.device.startswith("cuda"):
                model.to(self.device)
            return model
        except Exception as e:
            print(f"[WARN] Could not load {path} ({e}). Running in fallback mode.")
            return None

    def crop_driver_quadrant(self, cabin_crop: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Isolates the driver-side half of the windshield crop."""
        h, w = cabin_crop.shape[:2]
        if self.driver_side == DriverSide.RIGHT:
            # Right-hand drive: driver occupies horizontal 40% to 100%
            x1, x2 = int(w * 0.40), w
        else:
            # Left-hand drive: driver occupies horizontal 0% to 60%
            x1, x2 = 0, int(w * 0.60)

        driver_crop = cabin_crop[0:h, x1:x2]
        return driver_crop, [x1, 0, x2, h]

    def inspect_phone_usage(
        self,
        cabin_crop: np.ndarray,
        conf_thresh: float = 0.55,
    ) -> Tuple[bool, float, Optional[List[int]]]:
        """Evaluates whether the driver is actively using or holding a mobile phone.

        Returns:
            (is_violation, confidence_score, relative_bbox)
        """
        if cabin_crop.size == 0:
            return False, 0.0, None

        driver_roi, [qx1, qy1, _, _] = self.crop_driver_quadrant(cabin_crop)
        if driver_roi.size == 0:
            return False, 0.0, None

        # 1. Inference with trained phone/distraction model
        if self.model is not None:
            results = self.model(
                driver_roi,
                conf=conf_thresh,
                verbose=False,
                device=self.device,
                imgsz=320,
            )[0]

            # Expecting classes: 0 = 'phone_to_ear', 1 = 'phone_in_hand'
            if results.boxes is not None and len(results.boxes) > 0:
                best_box = results.boxes[0]
                conf = float(best_box.conf[0])
                b = best_box.xyxy[0].cpu().numpy()
                # Map coordinates relative to cabin_crop
                rel_box = [int(b[0] + qx1), int(b[1] + qy1), int(b[2] + qx1), int(b[3] + qy1)]
                return True, conf, rel_box

        # 2. Fallback when weights are not yet present
        return False, 0.0, None

