from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class Verdict(str, Enum):
    VIOLATION = "VIOLATION"
    COMPLIANT = "COMPLIANT"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class FramePrediction:
    is_violation: bool
    confidence: float
    frame_idx: int


class TemporalVotingEngine:
    def __init__(self, min_samples: int = 4, max_samples: int = 15, violation_threshold: float = 0.70):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.violation_threshold = violation_threshold
        self.track_buffers: Dict[int, List[FramePrediction]] = defaultdict(list)

    def record_prediction(self, track_id: int, is_violation: bool, confidence: float, frame_idx: int) -> None:
        buf = self.track_buffers[track_id]
        if len(buf) < self.max_samples:
            buf.append(FramePrediction(is_violation=is_violation, confidence=confidence, frame_idx=frame_idx))

    def compute_verdict(self, track_id: int) -> Tuple[Verdict, float, int]:
        buf = self.track_buffers.get(track_id, [])
        sample_count = len(buf)
        if sample_count < self.min_samples:
            return Verdict.UNCERTAIN, 0.0, sample_count

        total_weight = sum(p.confidence for p in buf)
        violation_weight = sum(p.confidence for p in buf if p.is_violation)
        if total_weight == 0.0:
            return Verdict.UNCERTAIN, 0.0, sample_count

        score = violation_weight / total_weight
        if score >= self.violation_threshold:
            return Verdict.VIOLATION, score, sample_count
        return Verdict.COMPLIANT, score, sample_count

    def clear_track(self, track_id: int) -> None:
        if track_id in self.track_buffers:
            del self.track_buffers[track_id]

