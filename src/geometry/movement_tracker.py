from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class RouteMovementTracker:
    """Tracks vehicle origins and destinations (Road A -> Road B) based on trajectory vectors."""

    def __init__(
        self,
        road_entry_name: str = "North Road",
        road_exit_name: str = "South Junction",
        min_displacement_px: float = 60.0,
    ):
        self.road_entry_name = road_entry_name
        self.road_exit_name = road_exit_name
        self.min_displacement_px = min_displacement_px

        # track_id -> trajectory points
        self.track_trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.recorded_movements = set()

        # Route matrix: vehicle_type -> {route_string: count}
        self.counts = {
            "Car": defaultdict(int),
            "Bike": defaultdict(int),
        }

    def update_position(self, track_id: int, bottom_center: Tuple[float, float]) -> None:
        """Records trajectory point for a vehicle."""
        self.track_trajectories[track_id].append(bottom_center)

    def finalize_route(self, track_id: int, vehicle_type: str = "Car") -> Optional[str]:
        """Calculates origin -> destination when vehicle leaves the scene or check zone."""
        if track_id in self.recorded_movements:
            return None

        points = self.track_trajectories.get(track_id, [])
        if len(points) < 4:
            return None

        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        dy = end_y - start_y
        dx = end_x - start_x

        route = None
        # Moving downwards (dy > 0) is Road A -> Road B
        # Moving upwards (dy < 0) is Road B -> Road A
        if dy > self.min_displacement_px:
            route = f"{self.road_entry_name} -> {self.road_exit_name}"
        elif dy < -self.min_displacement_px:
            route = f"{self.road_exit_name} -> {self.road_entry_name}"
        elif abs(dx) > self.min_displacement_px:
            # Lateral / Cross-street movement
            direction = "Eastbound" if dx > 0 else "Westbound"
            route = f"Side Street -> {direction}"

        if route:
            self.counts[vehicle_type][route] += 1
            self.recorded_movements.add(track_id)

        return route

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        return {
            "Car": dict(self.counts["Car"]),
            "Bike": dict(self.counts["Bike"]),
        }

