from typing import List, Optional, Tuple
import numpy as np


class RiderHead:
    def __init__(self, bbox: List[float], label: str, confidence: float):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.label = label  # "HELMET" or "NO_HELMET"
        self.confidence = confidence
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


class MotorcycleDriverSelector:
    def select_driver_head(
        self,
        heads: List[RiderHead],
        bike_motion_vector: Optional[Tuple[float, float]] = None,
        bike_bbox: Optional[List[float]] = None,
    ) -> Optional[RiderHead]:
        if not heads:
            return None
        if len(heads) == 1:
            return heads[0]

        # 1. Project candidate heads onto the travel vector
        if bike_motion_vector is not None:
            vx, vy = bike_motion_vector
            norm = np.hypot(vx, vy)
            if norm > 2.0:
                unit_v = (vx / norm, vy / norm)
                scores = [h.centroid[0] * unit_v[0] + h.centroid[1] * unit_v[1] for h in heads]
                return heads[int(np.argmax(scores))]

        # 2. Fallback when stationary: lower in frame (closer to camera)
        if bike_bbox is not None:
            return sorted(heads, key=lambda h: h.centroid[1], reverse=True)[0]

        return max(heads, key=lambda h: h.confidence)

