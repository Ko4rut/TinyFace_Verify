from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CroppedFace:
    image: np.ndarray
    landmarks: np.ndarray  # shape (5, 2)