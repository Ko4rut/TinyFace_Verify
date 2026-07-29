from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FaceAlignmentResult:
    """Result of aligning one cropped face to the reference template."""

    image: np.ndarray
    landmarks: np.ndarray
    transform_matrix: np.ndarray
