import torch
import numpy as np
from src.geometry.zone import CheckZone, VehicleZoneManager, ZoneState
from src.heuristics.rider_filter import MotorcycleDriverSelector, RiderHead
from src.heuristics.cabin_roi import CabinROIExtractor
from src.voting.temporal_voter import TemporalVotingEngine, Verdict

# 0. Test PyTorch & CUDA availability
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
print(f"[OK] PyTorch {torch.__version__} | CUDA available: {cuda_available} ({device_name})")

# 1. Test CheckZone Ray-Casting
zone = CheckZone("zone_main", [[100, 100], [500, 100], [550, 450], [50, 450]])
mgr = VehicleZoneManager(zone)
state, changed = mgr.update(track_id=1, bbox=[200, 200, 300, 350])
assert state == ZoneState.IN_ZONE and changed is True
print("[OK] Check-Zone geometry passed!")

# 2. Test Driver Disambiguation (Forward motion dy > 0)
selector = MotorcycleDriverSelector()
pillion = RiderHead(bbox=[150, 180, 200, 220], label="NO_HELMET", confidence=0.90)
driver = RiderHead(bbox=[150, 260, 200, 300], label="HELMET", confidence=0.94)
selected = selector.select_driver_head([pillion, driver], bike_motion_vector=(5.0, 40.0))
assert selected is driver and selected.label == "HELMET"
print("[OK] Driver vs. pillion disambiguation passed!")

# 3. Test Windshield CLAHE Extraction
extractor = CabinROIExtractor()
frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
crop, coords = extractor.extract_cabin(frame, [300, 200, 700, 500])
assert crop.shape[0] > 0 and crop.shape[1] > 0
print("[OK] Cabin ROI & CLAHE passed!")

# 4. Test Temporal Confidence Voting
voter = TemporalVotingEngine(min_samples=3, violation_threshold=0.70)
voter.record_prediction(track_id=10, is_violation=True, confidence=0.9, frame_idx=1)
voter.record_prediction(track_id=10, is_violation=True, confidence=0.85, frame_idx=2)
voter.record_prediction(track_id=10, is_violation=False, confidence=0.4, frame_idx=3)
verdict, score, count = voter.compute_verdict(track_id=10)
assert verdict == Verdict.VIOLATION
print(f"[OK] Temporal voting passed (Verdict: {verdict.value}, Score: {score:.2f})!")

print("\nAll core modules verified successfully.")


