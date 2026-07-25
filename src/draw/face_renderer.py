import cv2
import numpy as np

from src.detection.models import FaceDetection


class FaceRenderer:
    SELECTED_COLOR  = (0, 255, 0)
    LANDMARK_COLOR = (0, 0, 255)
    TEXT_COLOR = (0, 255, 0)
    OTHER_COLOR = (128, 128, 128)
    @classmethod
    def draw(
        cls,
        frame: np.ndarray,
        detections: list[FaceDetection],
        selected_face: FaceDetection | None = None,
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        for detection in detections:
            is_selected = detection is selected_face

            cls._draw_detection(
                frame=annotated_frame,
                detection=detection,
                is_selected=is_selected,
            )

            cls._draw_landmarks(
                frame=annotated_frame,
                detection=detection,
            )

        return annotated_frame

    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        detection: FaceDetection,
        is_selected: bool,
    ) -> None:
        x1, y1, x2, y2 = detection.bbox

        if is_selected:
            color = FaceRenderer.SELECTED_COLOR
            thickness = 3
            label = "TARGET"
        else:
            color = FaceRenderer.OTHER_COLOR
            thickness = 1
            label = "OTHER"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    
    # @classmethod
    # def _draw_bounding_box(
    #     cls,
    #     frame: np.ndarray,
    #     detection: FaceDetection,
    # ) -> None:
    #     x1, y1, x2, y2 = detection.bbox

    #     cv2.rectangle(
    #         frame,
    #         (x1, y1),
    #         (x2, y2),
    #         cls.BOX_COLOR,
    #         2,
    #     )

    #     cv2.putText(
    #         frame,
    #         f"{detection.confidence:.2f}",
    #         (x1, max(y1 - 10, 20)),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.6,
    #         cls.TEXT_COLOR,
    #         2,
    #     )

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