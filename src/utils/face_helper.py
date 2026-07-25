from src.detection.models import FaceDetection
import numpy as np
from src.detection.models import FaceDetection

def compute_width_height(face: FaceDetection) -> tuple[int,int]:
    x1, y1, x2, y2 = face.bbox
    width_face = abs(x1-x2)
    height_face =abs(y1-y2)
    return width_face, height_face

def crop_face(
    frame: np.ndarray,
    face: FaceDetection,
) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, face.bbox)

    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = frame[y1:y2, x1:x2].copy()

    if face_crop.size == 0:
        return None

    return face_crop