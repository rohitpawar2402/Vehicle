import os
import sys
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.secondary_detectors import SecondaryViolationDetector


def test_helmet_state_hysteresis_and_locking():
    detector = SecondaryViolationDetector(
        helmet_weights="models/yolov8n_helmet.pt",
        device="cpu",
    )

    track_id = 99
    # Simulate 3 frames confirming helmet
    l1 = detector._apply_hysteresis(track_id, instant_is_helmet=True)
    l2 = detector._apply_hysteresis(track_id, instant_is_helmet=True)
    l3 = detector._apply_hysteresis(track_id, instant_is_helmet=True)

    assert l3 == "HELMET", f"Expected HELMET on 3 consecutive detections, got {l3}"
    assert track_id in detector.locked_status, "track_id must be in locked_status"
    assert detector.locked_status[track_id] == "HELMET"

    # Now simulate momentary shadow/glare where heuristic/model returns False
    l4 = detector._apply_hysteresis(track_id, instant_is_helmet=False)
    l5 = detector._apply_hysteresis(track_id, instant_is_helmet=False)

    assert l4 == "HELMET", "Locked helmet status must remain HELMET despite momentary shadow"
    assert l5 == "HELMET", "Locked helmet status must remain HELMET despite momentary shadow"
    print("[OK] Temporal state hysteresis and permanent lock verified successfully!")


def test_bike_head_inspection():
    detector = SecondaryViolationDetector(
        helmet_weights="models/yolov8n_helmet.pt",
        device="cpu",
    )
    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    bike_bbox = [200.0, 300.0, 350.0, 600.0]

    heads = detector.inspect_motorcycle_helmets(
        frame=dummy_frame,
        bike_bbox=bike_bbox,
        track_id=101,
    )
    assert len(heads) >= 1, "Must return at least 1 rider head candidate"
    head = heads[0]
    assert head.label in ["HELMET", "NO_HELMET"]
    assert len(head.bbox) == 4
    print(f"[OK] Bike head inspection returned {head.label} with bbox {head.bbox}!")


if __name__ == "__main__":
    test_helmet_state_hysteresis_and_locking()
    test_bike_head_inspection()
    print("[ALL TESTS PASSED] Helmet detector & hysteresis tests verified!")

