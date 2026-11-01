from typing import List, Tuple
import cv2
import numpy as np


class CabinROIExtractor:
    def __init__(
        self,
        crop_ratios: Tuple[float, float, float, float] = (0.05, 0.60, 0.15, 0.85),
        apply_clahe: bool = True,
        clip_limit: float = 2.5,
        grid_size: Tuple[int, int] = (8, 8),
    ):
        self.ymin_r, self.ymax_r, self.xmin_r, self.xmax_r = crop_ratios
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def extract_cabin(self, frame: np.ndarray, car_bbox: List[float]) -> Tuple[np.ndarray, List[int]]:
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in car_bbox]
        w_car = x2 - x1
        h_car = y2 - y1

        cx1 = max(0, x1 + int(self.xmin_r * w_car))
        cx2 = min(w_frame, x1 + int(self.xmax_r * w_car))
        cy1 = max(0, y1 + int(self.ymin_r * h_car))
        cy2 = min(h_frame, y1 + int(self.ymax_r * h_car))

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return crop, [cx1, cy1, cx2, cy2]

        if self.apply_clahe:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            crop = cv2.cvtColor(cv2.merge((self.clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

        return crop, [cx1, cy1, cx2, cy2]

