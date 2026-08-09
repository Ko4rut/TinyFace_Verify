import cv2
import numpy as np

from src.detection.models import FaceDetection
from src.alignment.face_aligner import FaceAligner

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
        collected_count: int,
        required_frames: int,
        status: str,
        status_color: tuple[int, int, int],
        selected_face: FaceDetection | None = None,
        verification_result: bool | None = None,
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        for detection in detections:
            is_selected = detection is selected_face

            cls._draw_detection(
                frame=annotated_frame,
                detection=detection,
                is_selected=is_selected,
                verification_result=verification_result,
            )
            cls._draw_eye_landmarks(
                frame=annotated_frame,
                detection=detection,
            )
            
        cls._draw_info(
            frame=annotated_frame,  
            collected_count= collected_count,
            required_frames= required_frames,
            status= status,
            status_color=status_color,
        )

        return annotated_frame
    
    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        detection: FaceDetection,
        is_selected: bool,
        verification_result: bool | None = None,
    ) -> None:
        x1, y1, x2, y2 = detection.bbox

        if is_selected and verification_result is True:
            color = (0, 255, 0)
            thickness = 3
            label = "MATCHED"

        elif is_selected and verification_result is False:
            color = (0, 0, 255)
            thickness = 3
            label = "NOT MATCHED"

        elif is_selected:
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
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    
    @classmethod
    def _draw_eye_landmarks(
        cls,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> None:
        landmarks = np.asarray(
            detection.landmarks,
            dtype=np.float32,
        ).reshape(-1, 2)

        if len(landmarks) < 2:
            return

        left_eye = landmarks[0]
        right_eye = landmarks[1]

        left_eye_point = (
            int(left_eye[0]),
            int(left_eye[1]),
        )

        right_eye_point = (
            int(right_eye[0]),
            int(right_eye[1]),
        )

        # Vẽ hai landmark mắt
        for eye_point in (left_eye_point, right_eye_point):
            cv2.circle(
                frame,
                eye_point,
                3,
                (233, 196, 106),
                -1,
                cv2.LINE_AA,
            )

        # 1. Đường nối hai mắt thực tế
        cv2.line(
            frame,
            left_eye_point,
            right_eye_point,
            cls.LANDMARK_COLOR,
            2,
            cv2.LINE_AA,
        )

        # Tọa độ ngang đi qua tâm hai mắt
        eye_center_y = int(
            (left_eye[1] + right_eye[1]) / 2
        )

        x1, _, x2, _ = detection.bbox

        # 2. Đường ngang tham chiếu, chạy từ cạnh trái
        # đến cạnh phải của rectangle
        cv2.line(
            frame,
            (x1, eye_center_y),
            (x2, eye_center_y),
            cls.SELECTED_COLOR,
            1,
            cv2.LINE_AA,
        )

        angle = FaceAligner.calculate_eye_angle(
            left_eye=left_eye,
            right_eye=right_eye,
        )

        cv2.putText(
            frame,
            f"Angle: {angle:.1f} deg",
            (x1, max(20, eye_center_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cls.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
            
    @staticmethod
    def _draw_info(
        frame: np.ndarray,
        collected_count: int,
        required_frames: int,
        status: str,
        status_color: tuple[int, int, int],
    ):
        cv2.putText(
            frame,
            f"Collected: {collected_count}/{required_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
