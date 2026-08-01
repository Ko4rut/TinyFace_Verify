from dataclasses import dataclass

import numpy as np

from src.detection.models import FaceDetection


@dataclass
class CameraRenderState:
    detections: list[FaceDetection]
    selected_face: FaceDetection | None
    face_crop: np.ndarray | None
    crop_landmarks: np.ndarray | None

    original_landmarks: np.ndarray | None = None

    is_valid_sample: bool = False
    collected_count: int = 0
    required_frames: int = 0
    status: str = ""
    status_color: tuple[int, int, int] = (255, 255, 255)