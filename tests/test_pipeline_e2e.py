import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import cv2
import numpy as np
import supervision as sv

from src.geometry.zone import CheckZone, VehicleZoneManager, ZoneState
from src.dispatch.event_store import EventStore
from src.tracking.tracker import MultiClassVehicleTracker
from src.voting.temporal_voter import TemporalVotingEngine, Verdict


def test_event_store():
    test_db = "data/test_violations.db"
    test_snapshots = "data/test_snapshots"
    if os.path.exists(test_db):
        os.remove(test_db)

    store = EventStore(db_path=test_db, snapshot_dir=test_snapshots)
    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    crop_bbox = [200, 150, 400, 350]

    snapshot_path = store.record_violation(
        track_id=42,
        vehicle_type="Car",
        violation_type="NO_SEATBELT",
        confidence=0.88,
        sample_count=8,
        full_frame=dummy_frame,
        crop_bbox=crop_bbox,
    )

    assert os.path.exists(snapshot_path), f"Snapshot file not created at {snapshot_path}"
    saved_img = cv2.imread(snapshot_path)
    assert saved_img is not None, "Snapshot could not be read by OpenCV"

    stats = store.get_summary_stats()
    assert stats["total_violations"] == 1, f"Expected 1 violation, got {stats['total_violations']}"
    assert stats["by_type"].get("NO_SEATBELT") == 1
    print("[OK] EventStore SQLite and Snapshotter verified!")

    # Cleanup test snapshots
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)


def test_tracker_persistence():
    tracker = MultiClassVehicleTracker()
    frame_shape = (720, 1280)

    # Frame 1: Car at [100, 100, 200, 200]
    car_dets_1 = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200]]),
        confidence=np.array([0.90]),
        class_id=np.array([2]),
    )
    bike_dets_1 = sv.Detections.empty()
    tc1, tb1 = tracker.update(car_dets_1, bike_dets_1)
    assert tc1.tracker_id is not None and len(tc1.tracker_id) == 1
    initial_id = tc1.tracker_id[0]

    # Frame 2: Car moves slightly to [106, 106, 206, 206]
    car_dets_2 = sv.Detections(
        xyxy=np.array([[106, 106, 206, 206]]),
        confidence=np.array([0.88]),
        class_id=np.array([2]),
    )
    tc2, tb2 = tracker.update(car_dets_2, bike_dets_1)
    assert tc2.tracker_id is not None and len(tc2.tracker_id) == 1
    assert tc2.tracker_id[0] == initial_id, f"ID switch detected! Expected {initial_id}, got {tc2.tracker_id[0]}"
    print(f"[OK] MultiClassVehicleTracker persistent ID {initial_id} verified across frames!")


def test_zone_draw_and_motion():
    zone = CheckZone("zone_test", [[100, 100], [600, 100], [600, 500], [100, 500]])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    drawn = zone.draw(frame)
    assert drawn.shape == frame.shape, "Drawn frame dimensions do not match"

    mgr = VehicleZoneManager(zone)
    # Track vehicle moving downwards
    mgr.update(track_id=5, bbox=[200, 100, 300, 200])
    mgr.update(track_id=5, bbox=[200, 150, 300, 250])
    mgr.update(track_id=5, bbox=[200, 210, 300, 310])
    motion = mgr.get_motion_vector(track_id=5)
    assert motion is not None
    dx, dy = motion
    assert dy > 0, f"Expected positive forward dy motion, got dy={dy}"
    print(f"[OK] Zone drawing and motion vector (dx={dx:.1f}, dy={dy:.1f}) verified!")


if __name__ == "__main__":
    test_event_store()
    test_tracker_persistence()
    test_zone_draw_and_motion()
    print("\nAll Step 3 E2E unit tests passed successfully!")
