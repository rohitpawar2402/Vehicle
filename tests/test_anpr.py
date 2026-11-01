import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.detection.anpr import IndianANPR


def test_plate_regex_normalization():
    anpr = IndianANPR(use_gpu=False)

    # 1. State + District + Series + Number with character confusion
    # Letter 'O' in district -> '0', 'B' in number -> '8'
    raw_dirty = "MHO9AB12B4"
    cleaned = anpr._normalize_indian_plate(raw_dirty)
    assert cleaned == "MH09AB1284", f"Expected MH09AB1284, got {cleaned}"

    # 2. Bharat Series format: 22BH1234AA
    bh_dirty = "22BHI234AA"  # 'I' instead of '1'
    cleaned_bh = anpr._normalize_indian_plate(bh_dirty)
    assert cleaned_bh == "22BH1234AA", f"Expected 22BH1234AA, got {cleaned_bh}"

    print("[OK] Indian ANPR regex normalization verified successfully!")


def test_plate_crop_geometry():
    anpr = IndianANPR(use_gpu=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Car box
    car_bbox = [200, 200, 600, 500]
    crop, [px1, py1, px2, py2] = anpr.extract_plate_candidate(frame, car_bbox, vehicle_type="Car")
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert py1 >= 200 and py2 <= 500

    print("[OK] Plate candidate crop geometry verified!")


def test_empty_frame_handling():
    anpr = IndianANPR(use_gpu=False)
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    plate, conf = anpr.read_license_plate(empty_frame, [0, 0, 0, 0])
    assert plate == "UNREADABLE" and conf == 0.0

    print("[OK] Empty frame unreadable handling passed!")


if __name__ == "__main__":
    test_plate_regex_normalization()
    test_plate_crop_geometry()
    test_empty_frame_handling()
    print("\nAll ANPR unit tests passed successfully!")

