import cv2
import numpy as np

from src.detection.models import FaceDetection


class FaceRenderer:
    BOX_COLOR = (0, 255, 0)
    LANDMARK_COLOR = (0, 0, 255)
    TEXT_COLOR = (0, 255, 0)

    @classmethod
    def draw(
        cls,
        frame: np.ndarray,
        detections: list[FaceDetection],
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        for detection in detections:
            cls._draw_bounding_box(
                annotated_frame,
                detection,
            )
            cls._draw_landmarks(
                annotated_frame,
                detection,
            )

        return annotated_frame

    @classmethod
    def _draw_bounding_box(
        cls,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> None:
        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            cls.BOX_COLOR,
            2,
        )

        cv2.putText(
            frame,
            f"{detection.confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cls.TEXT_COLOR,
            2,
        )

    @classmethod
    def _draw_landmarks(
        cls,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> None:
        for landmark_x, landmark_y in detection.landmarks:
            cv2.circle(
                frame,
                (int(landmark_x), int(landmark_y)),
                3,
                cls.LANDMARK_COLOR,
                -1,
            )