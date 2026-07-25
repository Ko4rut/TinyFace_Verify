import cv2
import numpy as np

from src.detection.models import FaceDetection
from src.draw.face_renderer import FaceRenderer


class CameraRenderer:
    @staticmethod
    def render(
        frame: np.ndarray,
        detections: list[FaceDetection],
        collected_count: int,
        required_frames: int,
        status: str,
        status_color: tuple[int, int, int],
        selected_face: FaceDetection | None,

    ) -> np.ndarray:
        preview = frame.copy()

        preview = FaceRenderer.draw(
            preview,
            detections,
            selected_face=selected_face,
        )

        cv2.putText(
            preview,
            f"Collected: {collected_count}/{required_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            preview,
            status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        return preview