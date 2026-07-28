from dataclasses import dataclass

import numpy as np

from src.detection.models import FaceDetection


@dataclass
class CameraRenderState:
    detections: list[FaceDetection]
    selected_face: FaceDetection | None
    face_crop: np.ndarray | None
    crop_landmarks: np.ndarray | None
    is_valid_sample: bool

    collected_count: int
    required_frames: int

    status: str
    status_color: tuple[int, int, int]