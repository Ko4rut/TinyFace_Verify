import cv2
import numpy as np

from src.detection.models import FaceDetection
from src.draw.face_renderer import FaceRenderer
from src.draw.face_panel_renderer import FacePanelRenderer

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
        face_crop: np.ndarray | None,
        is_valid_sample: bool,
        face_landmarks: np.ndarray | None,

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
        face_panel = FacePanelRenderer.draw(
            frame_width=frame.shape[1],
            face_crop=face_crop,
            face_landmarks=face_landmarks,
            is_valid_sample=is_valid_sample,
            status=status,
        )
        return np.vstack((preview, face_panel))