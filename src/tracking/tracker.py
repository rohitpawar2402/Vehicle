import types
from typing import Optional, Tuple
import numpy as np
import supervision as sv
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class _ByteTrackerAdapter:
    """Wraps Ultralytics BYTETracker to provide an update_with_detections interface for sv.Detections."""

    def __init__(
        self,
        track_thresh: float = 0.20,
        track_buffer: int = 45,
        match_thresh: float = 0.75,
    ):
        args = types.SimpleNamespace(
            track_high_thresh=track_thresh,
            track_low_thresh=0.08,
            new_track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self.tracker = BYTETracker(args)

    def update_with_detections(self, detections: sv.Detections, frame_shape: Tuple[int, int] = (720, 1280)) -> sv.Detections:
        if len(detections) == 0:
            boxes = Boxes(torch.empty((0, 6)), frame_shape)
        else:
            xyxy = torch.from_numpy(detections.xyxy).float()
            conf = (
                torch.from_numpy(detections.confidence).float().unsqueeze(1)
                if detections.confidence is not None
                else torch.ones((len(detections), 1))
            )
            cls = (
                torch.from_numpy(detections.class_id).float().unsqueeze(1)
                if detections.class_id is not None
                else torch.zeros((len(detections), 1))
            )
            data = torch.cat([xyxy, conf, cls], dim=1)
            boxes = Boxes(data, frame_shape)

        tracked = self.tracker.update(boxes)
        if len(tracked) == 0:
            return sv.Detections.empty()

        tracked_arr = np.asarray(tracked)
        return sv.Detections(
            xyxy=tracked_arr[:, :4],
            confidence=tracked_arr[:, 5],
            class_id=tracked_arr[:, 6].astype(int),
            tracker_id=tracked_arr[:, 4].astype(int),
        )


class MultiClassVehicleTracker:
    """Maintains independent ByteTrack instances for cars and motorcycles to eliminate cross-class ID flipping."""

    def __init__(
        self,
        track_thresh: float = 0.20,
        track_buffer: int = 45,
        match_thresh: float = 0.75,
    ):
        if hasattr(sv, "ByteTrack"):
            self.car_tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
            )
            self.bike_tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
            )
        else:
            self.car_tracker = _ByteTrackerAdapter(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )
            self.bike_tracker = _ByteTrackerAdapter(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )

    def update(
        self,
        car_detections: sv.Detections,
        bike_detections: sv.Detections,
    ) -> Tuple[sv.Detections, sv.Detections]:
        """Updates trackers independently and returns tracked detections with persistent track_ids."""
        tracked_cars = self.car_tracker.update_with_detections(car_detections)
        tracked_bikes = self.bike_tracker.update_with_detections(bike_detections)
        return tracked_cars, tracked_bikes

