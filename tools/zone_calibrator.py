"""Interactive Road Calibration Tool (4-point Check Zone Quadrilateral).

Allows clicking 4 points on an image or video frame to define a road check zone polygon
and exports the coordinates to configs/zones.json.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple
import cv2
import numpy as np


class ZoneCalibrator:
    def __init__(self, window_name: str = "Zone Calibrator"):
        self.window_name = window_name
        self.points: List[Tuple[int, int]] = []
        self.current_frame: np.ndarray | None = None
        self.clean_frame: np.ndarray | None = None

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)} added: ({x}, {y})")
                self._draw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"Removed point: {removed}")
                self._draw()

    def _draw(self) -> None:
        if self.clean_frame is None:
            return
        display = self.clean_frame.copy()

        # Draw existing points
        for i, (px, py) in enumerate(self.points):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"P{i+1}:({px},{py})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw polygon lines if >= 2 points
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            is_closed = len(self.points) == 4
            cv2.polylines(display, [pts], isClosed=is_closed, color=(0, 255, 0), thickness=2)
            if is_closed:
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        # Help overlay
        info_text = (
            f"Points: {len(self.points)}/4 | "
            "Left-click: Add point | Right-click: Undo | 's': Save | 'c': Clear | 'q': Quit"
        )
        cv2.putText(
            display,
            info_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.current_frame = display
        cv2.imshow(self.window_name, self.current_frame)

    def calibrate(self, source: str, output_path: str = "configs/zones.json", camera_id: str = "camera_01") -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open source: {source}")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print(f"Error: Could not read frame from: {source}")
            return

        self.clean_frame = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._draw()

        print("\n=== Zone Calibration Instructions ===")
        print("1. Left-click to place 4 points of the check zone in clockwise or counter-clockwise order.")
        print("2. Right-click to undo the last point.")
        print("3. Press 'c' to clear all points.")
        print("4. Press 's' to save the 4-point polygon to configs/zones.json.")
        print("5. Press 'q' to quit without saving.\n")

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Exiting without saving.")
                break
            elif key == ord("c"):
                self.points.clear()
                self._draw()
                print("Points cleared.")
            elif key == ord("s"):
                if len(self.points) != 4:
                    print(f"Warning: Zone requires exactly 4 points (currently {len(self.points)}).")
                    continue

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                existing_data = {}
                if os.path.exists(output_path):
                    try:
                        with open(output_path, "r") as f:
                            existing_data = json.load(f)
                    except Exception:
                        existing_data = {}

                existing_data[camera_id] = {
                    "zone_id": f"zone_{camera_id}",
                    "polygon": [list(pt) for pt in self.points],
                    "color": [0, 255, 255],
                }

                with open(output_path, "w") as f:
                    json.dump(existing_data, f, indent=2)

                print(f"Successfully saved zone for '{camera_id}' to {output_path}: {self.points}")
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate 4-point check zone polygon.")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or RTSP stream URL")
    parser.add_argument("--output", type=str, default="configs/zones.json", help="Path to output JSON")
    parser.add_argument("--camera-id", type=str, default="camera_01", help="Camera identifier")
    args = parser.parse_args()

    calibrator = ZoneCalibrator()
    calibrator.calibrate(source=args.source, output_path=args.output, camera_id=args.camera_id)

