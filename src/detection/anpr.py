import os
import re
from typing import List, Optional, Tuple
import cv2
import numpy as np


class IndianANPR:
    """Extracts and normalizes Indian vehicle registration plates."""

    # Standard: State (2) + District (2) + Series (1-3) + Number (4) (e.g., MH09AB1234)
    STD_PATTERN = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$")
    # Bharat Series: Year (2) + BH + Number (4) + Series (1-2) (e.g., 22BH1234AA)
    BH_PATTERN = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")

    # Character confusion mapping
    CHAR_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
    DIGIT_TO_CHAR = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

    def __init__(self, use_gpu: bool = True, model_dir: str = "data/easyocr_models"):
        self.reader = None
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        os.environ["EASYOCR_MODULE_PATH"] = self.model_dir
        try:
            import easyocr

            self.reader = easyocr.Reader(
                ["en"],
                gpu=use_gpu,
                model_storage_directory=self.model_dir,
                user_network_directory=self.model_dir,
            )
            print("[INFO] EasyOCR initialized with CUDA support.")
        except Exception as e:
            print(f"[WARN] EasyOCR initialization failed ({e}). Running in fallback mode.")

    def extract_plate_candidate(
        self,
        frame: np.ndarray,
        vehicle_bbox: List[float],
        vehicle_type: str = "Car",
    ) -> Tuple[np.ndarray, List[int]]:
        """Crops the lower vehicle zone where number plates are mounted."""
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        h_veh, w_veh = y2 - y1, x2 - x1

        # Cars: plates mount at bottom 25-30%
        # Bikes: plates mount at bottom 35-40%
        y_ratio = 0.65 if vehicle_type.lower() == "car" else 0.60
        py1 = max(0, y1 + int(h_veh * y_ratio))
        py2 = min(h_frame, y2)
        px1 = max(0, x1 + int(w_veh * 0.15))
        px2 = min(w_frame, x2 - int(w_veh * 0.15))

        crop = frame[py1:py2, px1:px2]
        return crop, [px1, py1, px2, py2]

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Bilateral noise reduction and adaptive thresholding for dusty/glare conditions."""
        if plate_crop.size == 0:
            return plate_crop
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        # Smooth surface noise while preserving sharp character edges
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        # Enhance local contrast
        enhanced = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return enhanced

    def read_license_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: List[float],
        vehicle_type: str = "Car",
    ) -> Tuple[str, float]:
        """Extracts and validates registration plate string.

        Returns:
            (normalized_plate_string, confidence_score)
        """
        crop, _ = self.extract_plate_candidate(frame, vehicle_bbox, vehicle_type)
        if crop.size == 0 or self.reader is None:
            return "UNREADABLE", 0.0

        preprocessed = self.preprocess_plate(crop)
        results = self.reader.readtext(preprocessed, detail=1)
        if not results:
            # Fallback pass on raw color crop
            results = self.reader.readtext(crop, detail=1)

        if not results:
            return "UNREADABLE", 0.0

        # Combine detected text fragments
        raw_text = "".join([res[1] for res in results]).upper()
        raw_text = re.sub(r"[^A-Z0-9]", "", raw_text)  # Strip spaces/special chars
        avg_conf = float(np.mean([res[2] for res in results]))

        cleaned_plate = self._normalize_indian_plate(raw_text)
        return (
            cleaned_plate
            if cleaned_plate
            else (raw_text if len(raw_text) >= 6 else "UNREADABLE")
        ), avg_conf

    def _normalize_indian_plate(self, text: str) -> Optional[str]:
        """Validates and corrects OCR misreads using Indian RTO syntax rules."""
        if len(text) < 8 or len(text) > 11:
            return None

        # Check for Bharat Series: "BH" at positions 2:4
        chars = list(text)
        is_bharat = (
            chars[2] in ("B", "8")
            and chars[3] in ("H", "4")
            and (chars[0].isdigit() or chars[0] in self.CHAR_TO_DIGIT)
        )

        if is_bharat:
            # Format: YY BH #### AA
            # 0-1: Year (Digits)
            for i in range(2):
                if chars[i] in self.CHAR_TO_DIGIT:
                    chars[i] = self.CHAR_TO_DIGIT[chars[i]]
            # 2-3: "BH"
            chars[2] = "B"
            chars[3] = "H"
            # 4-7: 4 digits
            for i in range(4, min(8, len(chars))):
                if chars[i] in self.CHAR_TO_DIGIT:
                    chars[i] = self.CHAR_TO_DIGIT[chars[i]]
            # 8+: 1-2 letters
            for i in range(8, len(chars)):
                if chars[i] in self.DIGIT_TO_CHAR:
                    chars[i] = self.DIGIT_TO_CHAR[chars[i]]
        else:
            # Standard Indian plates: State (2 letters) + RTO (2 digits) + Series + Number (4 digits)
            # Indices 0-1: State Code (Letters, e.g., MH)
            for i in range(2):
                if chars[i] in self.DIGIT_TO_CHAR:
                    chars[i] = self.DIGIT_TO_CHAR[chars[i]]

            # Indices 2-3: RTO code (Digits, e.g., 09)
            for i in range(2, 4):
                if chars[i] in self.CHAR_TO_DIGIT:
                    chars[i] = self.CHAR_TO_DIGIT[chars[i]]

            # Last 4 characters: Serial Number (Digits, e.g., 1234)
            for i in range(len(chars) - 4, len(chars)):
                if chars[i] in self.CHAR_TO_DIGIT:
                    chars[i] = self.CHAR_TO_DIGIT[chars[i]]

            # Middle characters: Series (Letters)
            for i in range(4, len(chars) - 4):
                if chars[i] in self.DIGIT_TO_CHAR:
                    chars[i] = self.DIGIT_TO_CHAR[chars[i]]

        candidate = "".join(chars)
        if self.STD_PATTERN.match(candidate) or self.BH_PATTERN.match(candidate):
            return candidate
        return candidate if len(candidate) >= 8 else None
