from datetime import datetime
import os
import sqlite3
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class EventStore:
    """Manages persistent SQLite logging and high-resolution evidence snapshots."""

    def __init__(self, db_path: str = "data/violations.db", snapshot_dir: str = "data/snapshots"):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    license_plate TEXT DEFAULT 'UNKNOWN',
                    confidence REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    snapshot_path TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traffic_movements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    route TEXT NOT NULL
                )
            """)
            # Safe migration if table already exists without license_plate column
            cursor.execute("PRAGMA table_info(violations)")
            columns = [col[1] for col in cursor.fetchall()]
            if "license_plate" not in columns:
                cursor.execute("ALTER TABLE violations ADD COLUMN license_plate TEXT DEFAULT 'UNKNOWN'")
            conn.commit()

    def record_violation(
        self,
        track_id: int,
        vehicle_type: str,
        violation_type: str,
        confidence: float,
        sample_count: int,
        full_frame: np.ndarray,
        crop_bbox: List[float],
        license_plate: str = "UNKNOWN",
    ) -> str:
        """Saves high-res cropped snapshot and inserts record into SQLite."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Crop vehicle evidence with safety padding
        h, w = full_frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in crop_bbox]
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = full_frame[cy1:cy2, cx1:cx2]

        snapshot_name = f"{violation_type}_{vehicle_type}_id{track_id}_{file_timestamp}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)

        if crop.size > 0:
            cv2.imwrite(snapshot_path, crop)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO violations (timestamp, track_id, vehicle_type, violation_type, license_plate, confidence, sample_count, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (timestamp_str, track_id, vehicle_type, violation_type, license_plate, round(confidence, 3), sample_count, snapshot_path),
            )
            conn.commit()

        return snapshot_path

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns violation metrics for UI dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM violations")
            total_violations = cursor.fetchone()[0]

            cursor.execute("SELECT violation_type, COUNT(*) FROM violations GROUP BY violation_type")
            by_type = dict(cursor.fetchall())

        return {"total_violations": total_violations, "by_type": by_type}

    def record_movement(self, track_id: int, vehicle_type: str, route: str) -> None:
        """Logs a completed directional movement route for a vehicle."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO traffic_movements (timestamp, track_id, vehicle_type, route)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp_str, track_id, vehicle_type, route),
            )
            conn.commit()

    def get_movement_summary(self) -> Dict[str, Dict[str, int]]:
        """Returns aggregated road-to-road vehicle counts."""
        summary: Dict[str, Dict[str, int]] = {"Car": {}, "Bike": {}}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vehicle_type, route, COUNT(*) FROM traffic_movements GROUP BY vehicle_type, route"
            )
            for vtype, route, count in cursor.fetchall():
                if vtype not in summary:
                    summary[vtype] = {}
                summary[vtype][route] = count
        return summary


