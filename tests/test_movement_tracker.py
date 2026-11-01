import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.movement_tracker import RouteMovementTracker


def test_north_to_south_movement():
    tracker = RouteMovementTracker(road_entry_name="North Road", road_exit_name="South Junction", min_displacement_px=50.0)

    # Simulate vehicle moving from y=100 downwards to y=200
    positions = [(300, 100), (302, 130), (305, 170), (310, 200)]
    for pt in positions:
        tracker.update_position(track_id=1, bottom_center=pt)

    route = tracker.finalize_route(track_id=1, vehicle_type="Car")
    assert route == "North Road -> South Junction", f"Expected North -> South, got {route}"

    summary = tracker.get_summary()
    assert summary["Car"].get("North Road -> South Junction") == 1
    print("[OK] North -> South movement verified!")


def test_south_to_north_movement():
    tracker = RouteMovementTracker(road_entry_name="North Road", road_exit_name="South Junction", min_displacement_px=50.0)

    # Simulate motorcycle moving from y=500 upwards to y=350
    positions = [(400, 500), (401, 450), (398, 400), (395, 350)]
    for pt in positions:
        tracker.update_position(track_id=2, bottom_center=pt)

    route = tracker.finalize_route(track_id=2, vehicle_type="Bike")
    assert route == "South Junction -> North Road", f"Expected South -> North, got {route}"

    summary = tracker.get_summary()
    assert summary["Bike"].get("South Junction -> North Road") == 1
    print("[OK] South -> North movement verified!")


def test_lateral_movement():
    tracker = RouteMovementTracker(min_displacement_px=50.0)

    # Simulate vehicle moving horizontally to the right (Eastbound)
    positions = [(100, 300), (130, 301), (170, 302), (210, 303)]
    for pt in positions:
        tracker.update_position(track_id=3, bottom_center=pt)

    route = tracker.finalize_route(track_id=3, vehicle_type="Car")
    assert route == "Side Street -> Eastbound", f"Expected Eastbound, got {route}"
    print("[OK] Lateral movement verified!")


def test_insufficient_points_or_displacement():
    tracker = RouteMovementTracker(min_displacement_px=50.0)

    # Only 2 points
    tracker.update_position(track_id=4, bottom_center=(200, 200))
    tracker.update_position(track_id=4, bottom_center=(200, 300))
    route = tracker.finalize_route(track_id=4, vehicle_type="Car")
    assert route is None, "Expected None for fewer than 4 points"

    # 4 points, but displacement is too small (< 50px)
    tracker.update_position(track_id=5, bottom_center=(200, 200))
    tracker.update_position(track_id=5, bottom_center=(200, 210))
    tracker.update_position(track_id=5, bottom_center=(200, 215))
    tracker.update_position(track_id=5, bottom_center=(200, 220))
    route = tracker.finalize_route(track_id=5, vehicle_type="Car")
    assert route is None, "Expected None for displacement below min_displacement_px"
    print("[OK] Threshold gating verified!")


def test_idempotent_recording():
    tracker = RouteMovementTracker(min_displacement_px=50.0)
    for y in [100, 150, 200, 250]:
        tracker.update_position(track_id=6, bottom_center=(300, y))

    route1 = tracker.finalize_route(track_id=6, vehicle_type="Car")
    route2 = tracker.finalize_route(track_id=6, vehicle_type="Car")

    assert route1 is not None
    assert route2 is None, "Second finalize_route call must return None"
    assert tracker.counts["Car"][route1] == 1, "Must not double count on multiple calls"
    print("[OK] Idempotent route recording verified!")


if __name__ == "__main__":
    test_north_to_south_movement()
    test_south_to_north_movement()
    test_lateral_movement()
    test_insufficient_points_or_displacement()
    test_idempotent_recording()
    print("[ALL TESTS PASSED] RouteMovementTracker fully verified!")

