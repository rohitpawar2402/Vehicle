from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ZoneState(str, Enum):
    APPROACHING = "APPROACHING"
    IN_ZONE = "IN_ZONE"
    EVALUATED = "EVALUATED"
    EXITED = "EXITED"


class CheckZone:
    def __init__(self, zone_id: str, points: List[List[int]], color: Tuple[int, int, int] = (0, 255, 255)):
        self.zone_id = zone_id
        self.points = np.array(points, dtype=np.int32)
        if len(self.points) < 3:
            raise ValueError("Check zone must have at least 3 points.")
        self.color = color

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Ray-Casting algorithm to check if point is inside polygon."""
        dist = cv2.pointPolygonTest(self.points, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0

    @staticmethod
    def get_contact_point(bbox: List[float]) -> Tuple[float, float]:
        """Calculates tire-road contact point [x1, y1, x2, y2] to minimize camera parallax."""
        x1, _, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def draw(self, frame: np.ndarray, alpha: float = 0.20) -> np.ndarray:
        """Draws semi-transparent polygon check zone with solid border on frame."""
        output = frame.copy()
        overlay = frame.copy()
        pts = self.points.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color=self.color)
        cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)
        cv2.polylines(output, [pts], isClosed=True, color=self.color, thickness=2, lineType=cv2.LINE_AA)
        
        # Draw zone label at first point
        lbl_x, lbl_y = int(self.points[0][0]), int(self.points[0][1])
        cv2.putText(output, self.zone_id, (lbl_x + 5, lbl_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        return output


class VehicleZoneManager:
    def __init__(self, zone: CheckZone):
        self.zone = zone
        self.track_states: Dict[int, ZoneState] = {}
        self.trajectory_history: Dict[int, List[Tuple[float, float]]] = {}

    def update(self, track_id: int, bbox: List[float]) -> Tuple[ZoneState, bool]:
        contact_point = self.zone.get_contact_point(bbox)
        if track_id not in self.trajectory_history:
            self.trajectory_history[track_id] = []
        self.trajectory_history[track_id].append(contact_point)

        is_inside = self.zone.contains_point(contact_point)
        current_state = self.track_states.get(track_id, ZoneState.APPROACHING)
        state_changed = False

        if current_state == ZoneState.APPROACHING and is_inside:
            current_state = ZoneState.IN_ZONE
            state_changed = True
        elif current_state == ZoneState.IN_ZONE and not is_inside:
            current_state = ZoneState.EXITED
            state_changed = True

        self.track_states[track_id] = current_state
        return current_state, state_changed

    def get_motion_vector(self, track_id: int, window: int = 5) -> Optional[Tuple[float, float]]:
        """Calculates trajectory motion vector (dx, dy) across the recent position history."""
        history = self.trajectory_history.get(track_id, [])
        if len(history) < 2:
            return None
        start = history[-min(window, len(history))]
        end = history[-1]
        return (end[0] - start[0], end[1] - start[1])

    def mark_evaluated(self, track_id: int) -> None:
        self.track_states[track_id] = ZoneState.EVALUATED
