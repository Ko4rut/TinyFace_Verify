from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray