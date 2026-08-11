from dataclasses import dataclass

import numpy as np

from src.alignment.models import FaceAlignmentResult
from src.selection.models import FaceSelectionResult
from src.detection.models import FaceDetection

@dataclass
class FrameAnalysis:
    detections: list[FaceDetection]
    selection: FaceSelectionResult
    selected_face: FaceDetection | None
    alignment_result: FaceAlignmentResult | None
    is_sample_ready: bool