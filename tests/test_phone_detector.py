import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.detection.phone_detector import DriverSide, MobilePhoneDetector


def test_driver_quadrant_cropping():
    detector_rhd = MobilePhoneDetector(driver_side=DriverSide.RIGHT)
    detector_lhd = MobilePhoneDetector(driver_side=DriverSide.LEFT)

    # Fake cabin crop (height=100, width=200, channels=3)
    cabin_crop = np.zeros((100, 200, 3), dtype=np.uint8)

    # Right-Hand Drive (India / UK)
    rhd_crop, [rx1, ry1, rx2, ry2] = detector_rhd.crop_driver_quadrant(cabin_crop)
    assert rx1 == 80 and rx2 == 200, f"Expected RHD [80, 200], got [{rx1}, {rx2}]"
    assert rhd_crop.shape == (100, 120, 3)

    # Left-Hand Drive (US / EU)
    lhd_crop, [lx1, ly1, lx2, ly2] = detector_lhd.crop_driver_quadrant(cabin_crop)
    assert lx1 == 0 and lx2 == 120, f"Expected LHD [0, 120], got [{lx1}, {lx2}]"
    assert lhd_crop.shape == (100, 120, 3)

    print("[OK] Driver quadrant isolation (RHD & LHD) passed!")


def test_phone_fallback_and_empty():
    detector = MobilePhoneDetector(weights_path="models/nonexistent.pt")

    # Empty crop
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    is_viol, conf, box = detector.inspect_phone_usage(empty_frame)
    assert is_viol is False and conf == 0.0 and box is None

    # Normal crop with missing model
    cabin_crop = np.zeros((100, 200, 3), dtype=np.uint8)
    is_viol, conf, box = detector.inspect_phone_usage(cabin_crop)
    assert is_viol is False and conf == 0.0 and box is None

    print("[OK] Phone detector fallback and empty crop handling passed!")


if __name__ == "__main__":
    test_driver_quadrant_cropping()
    test_phone_fallback_and_empty()
    print("\nAll Mobile Phone Detector unit tests passed successfully!")

