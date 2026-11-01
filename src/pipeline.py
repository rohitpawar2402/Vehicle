import argparse
import json
import os
from typing import List, Optional
import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from src.detection.anpr import IndianANPR
from src.detection.phone_detector import DriverSide, MobilePhoneDetector
from src.detection.secondary_detectors import SecondaryViolationDetector
from src.dispatch.event_store import EventStore
from src.geometry import CheckZone, RouteMovementTracker, VehicleZoneManager, ZoneState
from src.heuristics.cabin_roi import CabinROIExtractor
from src.heuristics.rider_filter import MotorcycleDriverSelector, RiderHead
from src.ingestion.stream_buffer import StreamBuffer
from src.tracking.tracker import MultiClassVehicleTracker
from src.voting.temporal_voter import TemporalVotingEngine, Verdict


class TrafficViolationPipeline:
    """Orchestrates end-to-end detection, tracking, zone triggers, and violation logging."""

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        helmet_weights: str = "models/yolov8n_helmet.pt",
        seatbelt_weights: str = "models/yolov8s_seatbelt.pt",
        phone_weights: str = "models/yolov8s_phone.pt",
        driver_side: DriverSide = DriverSide.RIGHT,
        zone_polygon: Optional[List[List[int]]] = None,
        camera_id: Optional[str] = None,
        device: str = "cuda:0",
        db_path: str = "data/violations.db",
        snapshot_dir: str = "data/snapshots",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing Pipeline on: {self.device}")

        # 1. Primary Model: FP16 half precision for RTX 3050 speed & VRAM efficiency
        self.model = YOLO(model_path)
        if self.device.startswith("cuda"):
            self.model.to(self.device)

        # 2. Geometry Check-Zone (Load from config or default quad)
        zone_id = camera_id or "zone_main"
        if zone_polygon is None and os.path.exists("configs/zones.json"):
            try:
                with open("configs/zones.json", "r") as f:
                    zdata = json.load(f)
                    # Check requested camera_id first
                    if camera_id and camera_id in zdata and "polygon" in zdata[camera_id]:
                        zone_polygon = zdata[camera_id]["polygon"]
                        zone_id = zdata[camera_id].get("zone_id", camera_id)
                    else:
                        for k in ["cam_bike", "cam01", "camera_01"]:
                            if k in zdata and "polygon" in zdata[k]:
                                zone_polygon = zdata[k]["polygon"]
                                zone_id = zdata[k].get("zone_id", k)
                                break
                    if zone_polygon is None and zdata:
                        first_k = next(iter(zdata))
                        zone_polygon = zdata[first_k].get("polygon")
                        zone_id = zdata[first_k].get("zone_id", first_k)
            except Exception as e:
                print(f"[WARN] Failed loading configs/zones.json: {e}")

        if zone_polygon is None:
            zone_polygon = [[150, 200], [1130, 200], [1200, 680], [80, 680]]
        self.check_zone = CheckZone(zone_id, zone_polygon)
        self.zone_manager = VehicleZoneManager(self.check_zone)

        # 3. Secondary Detectors for Helmet and Seatbelt inspection
        self.secondary_detector = SecondaryViolationDetector(
            helmet_weights=helmet_weights,
            seatbelt_weights=seatbelt_weights,
            device=self.device,
            half=True,
        )

        # 4. Mobile Phone Usage Detector (Driver distraction)
        self.phone_detector = MobilePhoneDetector(
            weights_path=phone_weights,
            driver_side=driver_side,
            device=self.device,
            half=True,
        )

        # 5. Trackers & Analytics
        self.tracker = MultiClassVehicleTracker(track_thresh=0.20, track_buffer=45, match_thresh=0.75)
        self.route_tracker = RouteMovementTracker(road_entry_name="North Road", road_exit_name="South Junction")
        self.voting_engine = TemporalVotingEngine(min_samples=4, violation_threshold=0.65)
        self.event_store = EventStore(db_path=db_path, snapshot_dir=snapshot_dir)

        # 6. Automatic Number Plate Recognition (ANPR)
        self.anpr = IndianANPR(use_gpu=(self.device.startswith("cuda")))

        # 7. Heuristics Engines
        self.cabin_extractor = CabinROIExtractor(apply_clahe=True)
        self.rider_selector = MotorcycleDriverSelector()

        # 8. UI Counters & State
        self.total_cars = set()
        self.total_bikes = set()
        self.prev_active_car_ids = set()
        self.prev_active_bike_ids = set()
        self.logged_violations = set()
        self.logged_phone_violations = set()
        self.logged_triple_riding = set()

    def process_stream(
        self,
        source: str = "0",
        output_path: Optional[str] = None,
        show: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        is_live = str(source).isdigit() or str(source).startswith("rtsp://") or str(source).startswith("http://")
        stream_buf: Optional[StreamBuffer] = None
        cap: Optional[cv2.VideoCapture] = None

        if is_live:
            print(f"[INFO] Connecting decoupled RTSP/live stream buffer to: {source}")
            stream_buf = StreamBuffer(source=source, buffer_size=2).start()
            w, h, fps = stream_buf.width, stream_buf.height, stream_buf.fps
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open stream source: {source}")
                return
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        print(f"[INFO] Processing stream... ({'Press q to stop' if show else 'Running headless'})")

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            if is_live and stream_buf:
                ret, frame = stream_buf.read(timeout=1.0)
            elif cap:
                ret, frame = cap.read()
            else:
                break

            if not ret or frame is None:
                break
            frame_idx += 1

            # --- A. Primary Detection (Persons: 0, Bicycles: 1, Cars: 2, Motorcycles: 3) ---
            results = self.model(
                frame,
                classes=[0, 1, 2, 3],
                conf=0.18,
                iou=0.45,
                verbose=False,
                device=self.device,
                half=(self.device.startswith("cuda")),
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # 1. Car detections
            car_dets = detections[detections.class_id == 2]

            # 2. Pool two-wheelers (motorcycles + scooters/bicycles)
            bike_mask = np.isin(detections.class_id, [1, 3])
            bike_dets = detections[bike_mask]

            # 3. Merge Rider (Person) into Motorcycle Box so the box covers head-to-wheel
            person_dets = detections[detections.class_id == 0]
            if len(bike_dets) > 0 and len(person_dets) > 0:
                for i, b in enumerate(bike_dets.xyxy):
                    for p in person_dets.xyxy:
                        # If person horizontally aligns with the bike (rider sitting on it)
                        overlap_x = max(0.0, min(b[2], p[2]) - max(b[0], p[0]))
                        if overlap_x > 0.35 * (b[2] - b[0]):
                            # Expand box to include rider's head and body
                            bike_dets.xyxy[i][0] = min(b[0], p[0])
                            bike_dets.xyxy[i][1] = min(b[1], p[1])  # Reaches top of helmet!
                            bike_dets.xyxy[i][2] = max(b[2], p[2])
                            bike_dets.xyxy[i][3] = max(b[3], p[3])
                            break

            # 4. Convert unmerged roadway persons (occluded motorcycle riders) into bike candidates
            if len(person_dets) > 0:
                unmerged = []
                for p in person_dets.xyxy:
                    overlaps_existing = False
                    if len(bike_dets) > 0:
                        for b in bike_dets.xyxy:
                            ox = max(0.0, min(b[2], p[2]) - max(b[0], p[0]))
                            oy = max(0.0, min(b[3], p[3]) - max(b[1], p[1]))
                            if ox > 0.30 * (p[2] - p[0]) and oy > 0.30 * (p[3] - p[1]):
                                overlaps_existing = True
                                break
                    if not overlaps_existing and len(car_dets) > 0:
                        for c in car_dets.xyxy:
                            ox = max(0.0, min(c[2], p[2]) - max(c[0], p[0]))
                            oy = max(0.0, min(c[3], p[3]) - max(c[1], p[1]))
                            if ox > 0.30 * (p[2] - p[0]) and oy > 0.30 * (p[3] - p[1]):
                                overlaps_existing = True
                                break
                    if not overlaps_existing:
                        pw = p[2] - p[0]
                        ph = p[3] - p[1]
                        if ph / max(1.0, pw) >= 0.90:  # Typical upright/seated rider aspect ratio
                            expanded_box = [p[0], p[1], p[2], min(h, p[3] + ph * 0.40)]
                            unmerged.append(expanded_box)

                if unmerged:
                    extra_boxes = np.array(unmerged)
                    extra_conf = np.ones(len(unmerged)) * 0.70
                    extra_cls = np.ones(len(unmerged)) * 3
                    if len(bike_dets) > 0:
                        bike_dets = sv.Detections(
                            xyxy=np.vstack([bike_dets.xyxy, extra_boxes]),
                            confidence=np.concatenate([bike_dets.confidence, extra_conf]),
                            class_id=np.concatenate([bike_dets.class_id, extra_cls]),
                        )
                    else:
                        bike_dets = sv.Detections(
                            xyxy=extra_boxes,
                            confidence=extra_conf,
                            class_id=extra_cls,
                        )

            person_boxes = person_dets.xyxy.tolist() if len(person_dets) > 0 else []
            active_heads: List[RiderHead] = []

            # --- B. Class-Separated Tracking ---
            tracked_cars, tracked_bikes = self.tracker.update(car_dets, bike_dets)
            current_car_ids = set()
            current_bike_ids = set()

            # --- C. Process Tracked Cars ---
            if tracked_cars.tracker_id is not None:
                for bbox, track_id in zip(tracked_cars.xyxy, tracked_cars.tracker_id):
                    self.total_cars.add(track_id)
                    current_car_ids.add(track_id)

                    # Update trajectory for origin-destination route tracking
                    bx1, by1, bx2, by2 = bbox.tolist()
                    bottom_center = ((bx1 + bx2) / 2.0, by2)
                    self.route_tracker.update_position(track_id, bottom_center)

                    state, changed = self.zone_manager.update(track_id, bbox.tolist())

                    if state == ZoneState.IN_ZONE:
                        # Extract cabin ROI with CLAHE glare suppression
                        cabin_crop, _ = self.cabin_extractor.extract_cabin(frame, bbox.tolist())

                        # 1. Seatbelt Inspection
                        is_seatbelt_violation, sb_conf = self.secondary_detector.inspect_car_seatbelt(cabin_crop)
                        if sb_conf > 0.40:
                            self.voting_engine.record_prediction(track_id, is_seatbelt_violation, sb_conf, frame_idx)

                        # 2. Mobile Phone Usage Inspection
                        is_phone_violation, phone_conf, _ = self.phone_detector.inspect_phone_usage(cabin_crop)
                        if is_phone_violation and phone_conf > 0.60 and track_id not in self.logged_phone_violations:
                            plate_number, _ = self.anpr.read_license_plate(frame, bbox.tolist(), vehicle_type="Car")
                            self.event_store.record_violation(
                                track_id, "Car", "PHONE_USAGE", phone_conf, 1, frame, bbox.tolist(), license_plate=plate_number
                            )
                            self.logged_phone_violations.add(track_id)

                    elif state == ZoneState.EXITED:
                        route = self.route_tracker.finalize_route(track_id, "Car")
                        if route:
                            self.event_store.record_movement(track_id, "Car", route)

                        if track_id not in self.logged_violations:
                            verdict, score, samples = self.voting_engine.compute_verdict(track_id)
                            if verdict == Verdict.VIOLATION:
                                plate_number, _ = self.anpr.read_license_plate(frame, bbox.tolist(), vehicle_type="Car")
                                self.event_store.record_violation(
                                    track_id, "Car", "NO_SEATBELT", score, samples, frame, bbox.tolist(), license_plate=plate_number
                                )
                                self.logged_violations.add(track_id)
                            self.voting_engine.clear_track(track_id)
                            self.zone_manager.mark_evaluated(track_id)

            # --- D. Process Tracked Motorcycles ---
            if tracked_bikes.tracker_id is not None:
                for bbox, track_id in zip(tracked_bikes.xyxy, tracked_bikes.tracker_id):
                    self.total_bikes.add(track_id)
                    current_bike_ids.add(track_id)

                    # Update trajectory for origin-destination route tracking
                    bx1, by1, bx2, by2 = bbox.tolist()
                    bottom_center = ((bx1 + bx2) / 2.0, by2)
                    self.route_tracker.update_position(track_id, bottom_center)

                    # Always inspect rider heads for visible bounding boxes
                    heads = self.secondary_detector.inspect_motorcycle_helmets(
                        frame, bbox.tolist(), person_boxes=person_boxes, track_id=track_id
                    )
                    active_heads.extend(heads)

                    state, changed = self.zone_manager.update(track_id, bbox.tolist())

                    if state == ZoneState.IN_ZONE:
                        # Triple-Riding Check (3+ riders detected on a motorcycle)
                        if len(heads) >= 3 and track_id not in self.logged_triple_riding:
                            plate_number, _ = self.anpr.read_license_plate(frame, bbox.tolist(), vehicle_type="Bike")
                            self.event_store.record_violation(
                                track_id, "Bike", "TRIPLE_RIDING", 0.90, len(heads), frame, bbox.tolist(), license_plate=plate_number
                            )
                            self.logged_triple_riding.add(track_id)

                        motion = self.zone_manager.get_motion_vector(track_id)
                        driver = self.rider_selector.select_driver_head(
                            heads, bike_motion_vector=motion, bike_bbox=bbox.tolist()
                        )
                        if driver:
                            is_violation = (driver.label == "NO_HELMET")
                            self.voting_engine.record_prediction(track_id, is_violation, driver.confidence, frame_idx)

                    elif state == ZoneState.EXITED:
                        route = self.route_tracker.finalize_route(track_id, "Bike")
                        if route:
                            self.event_store.record_movement(track_id, "Bike", route)

                        if track_id not in self.logged_violations:
                            verdict, score, samples = self.voting_engine.compute_verdict(track_id)
                            if verdict == Verdict.VIOLATION:
                                plate_number, _ = self.anpr.read_license_plate(frame, bbox.tolist(), vehicle_type="Bike")
                                self.event_store.record_violation(
                                    track_id, "Bike", "NO_HELMET", score, samples, frame, bbox.tolist(), license_plate=plate_number
                                )
                                self.logged_violations.add(track_id)
                            self.voting_engine.clear_track(track_id)
                            self.zone_manager.mark_evaluated(track_id)

            # Finalize routes for tracks that left the frame without triggering EXITED
            for missing_id in (self.prev_active_car_ids - current_car_ids):
                route = self.route_tracker.finalize_route(missing_id, "Car")
                if route:
                    self.event_store.record_movement(missing_id, "Car", route)
            for missing_id in (self.prev_active_bike_ids - current_bike_ids):
                route = self.route_tracker.finalize_route(missing_id, "Bike")
                if route:
                    self.event_store.record_movement(missing_id, "Bike", route)
            self.prev_active_car_ids = current_car_ids
            self.prev_active_bike_ids = current_bike_ids

            # --- E. Render Visual Annotations ---
            display = self.check_zone.draw(frame, alpha=0.20)
            self._render_hud(display)

            # Draw vehicle bounding boxes
            for tracked, color, label in [
                (tracked_cars, (255, 180, 0), "Car"),
                (tracked_bikes, (0, 200, 255), "Bike"),
            ]:
                if tracked.tracker_id is not None:
                    for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            display,
                            f"{label} #{tid}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

            # Draw detected rider heads / helmets
            for head in active_heads:
                hx1, hy1, hx2, hy2 = [int(v) for v in head.bbox]
                h_color = (0, 255, 0) if head.label == "HELMET" else (0, 0, 255)
                cv2.rectangle(display, (hx1, hy1), (hx2, hy2), h_color, 2)
                cv2.putText(
                    display,
                    head.label,
                    (hx1, max(18, hy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    h_color,
                    2,
                )

            if writer:
                writer.write(display)

            if show:
                try:
                    cv2.imshow("CCTV Traffic Violation Enforcement", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("[INFO] OpenCV GUI backend not available. Continuing in headless mode...")
                    show = False

        if stream_buf:
            stream_buf.stop()
        if cap:
            cap.release()
        if writer:
            writer.release()
            # Automatically transcode to HTML5 browser-compatible H.264 (yuv420p)
            if output_path and os.path.exists(output_path):
                try:
                    import av
                    temp_h264 = output_path + ".h264.mp4"
                    input_container = av.open(output_path)
                    in_video = input_container.streams.video[0]
                    output_container = av.open(temp_h264, mode="w")
                    out_video = output_container.add_stream("libx264", rate=in_video.average_rate or 30)
                    out_video.width, out_video.height = in_video.width, in_video.height
                    out_video.pix_fmt = "yuv420p"
                    out_video.options = {"crf": "23", "preset": "fast"}
                    for f in input_container.decode(in_video):
                        for pkt in out_video.encode(f):
                            output_container.mux(pkt)
                    for pkt in out_video.encode():
                        output_container.mux(pkt)
                    input_container.close()
                    output_container.close()
                    os.replace(temp_h264, output_path)
                    print(f"[INFO] Video saved with HTML5 browser-compatible H.264 codec: {output_path}")
                except Exception as e:
                    print(f"[WARN] H.264 web transcode skipped ({e}). Video remains in mp4v format.")

        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"[COMPLETE] Processed {frame_idx} frames. Violations logged to {self.event_store.db_path}")

    def _render_hud(self, frame: np.ndarray) -> None:
        """Renders live violation counters, compliance rate, and road movement matrix."""
        stats = self.event_store.get_summary_stats()
        total_vehicles = len(self.total_cars) + len(self.total_bikes)
        total_viol = stats["total_violations"]
        compliance_pct = (
            100.0 if total_vehicles == 0 else max(0.0, ((total_vehicles - total_viol) / total_vehicles) * 100.0)
        )

        h, w = frame.shape[:2]

        # Top-left dashboard banner (Violations & Compliance)
        cv2.rectangle(frame, (10, 10), (420, 115), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, 10), (420, 115), (100, 100, 100), 1)

        cv2.putText(
            frame,
            f"Cars: {len(self.total_cars)}  |  Bikes: {len(self.total_bikes)}",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Violations: {total_viol}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Compliance Rate: {compliance_pct:.1f}%",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        # Top-right dashboard banner (Road-to-Road Directional Movement Counts)
        summary = self.route_tracker.get_summary()
        c_counts = summary["Car"]
        b_counts = summary["Bike"]

        ns_name = f"{self.route_tracker.road_entry_name} -> {self.route_tracker.road_exit_name}"
        sn_name = f"{self.route_tracker.road_exit_name} -> {self.route_tracker.road_entry_name}"

        c_ns = c_counts.get(ns_name, 0)
        c_sn = c_counts.get(sn_name, 0)
        b_ns = b_counts.get(ns_name, 0)
        b_sn = b_counts.get(sn_name, 0)

        box_w = 440
        x_start = max(10, w - box_w - 10)
        x_end = w - 10

        cv2.rectangle(frame, (x_start, 10), (x_end, 115), (25, 25, 25), -1)
        cv2.rectangle(frame, (x_start, 10), (x_end, 115), (100, 100, 100), 1)

        cv2.putText(
            frame,
            "Movement Flow (Origin -> Dest)",
            (x_start + 15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Cars:  N->S: {c_ns}  |  S->N: {c_sn}",
            (x_start + 15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Bikes: N->S: {b_ns}  |  S->N: {b_sn}",
            (x_start + 15, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Traffic Violation Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Webcam index, video path, or RTSP URL")
    parser.add_argument("--output", type=str, default=None, help="Optional output video save path (.mp4)")
    parser.add_argument("--camera-id", type=str, default=None, help="Camera identifier for zone lookup (e.g., cam_bike)")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Primary vehicle detector weights path")
    parser.add_argument("--helmet-weights", type=str, default="models/yolov8n_helmet.pt", help="Secondary helmet detector weights")
    parser.add_argument("--seatbelt-weights", type=str, default="models/yolov8s_seatbelt.pt", help="Secondary seatbelt detector weights")
    parser.add_argument("--phone-weights", type=str, default="models/yolov8s_phone.pt", help="Secondary phone usage detector weights")
    parser.add_argument("--driver-side", type=str, choices=["RIGHT", "LEFT"], default="RIGHT", help="Driver side (RIGHT for India/UK, LEFT for US/EU)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    pipeline = TrafficViolationPipeline(
        model_path=args.model,
        helmet_weights=args.helmet_weights,
        seatbelt_weights=args.seatbelt_weights,
        phone_weights=args.phone_weights,
        driver_side=DriverSide(args.driver_side),
        camera_id=args.camera_id,
    )
    pipeline.process_stream(
        source=args.source,
        output_path=args.output,
        show=not args.headless,
        max_frames=args.max_frames,
    )

