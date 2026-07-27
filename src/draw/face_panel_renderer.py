import cv2
import numpy as np


class FacePanelRenderer:
    PANEL_HEIGHT = 160
    PADDING = 15
    CROP_SIZE = 130

    VALID_COLOR = (0, 255, 0)
    INVALID_COLOR = (0, 165, 255)
    EMPTY_COLOR = (100, 100, 100)
    TEXT_COLOR = (230, 230, 230)
    BACKGROUND_COLOR = (30, 30, 30)

    @classmethod
    def draw(
        cls,
        frame_width: int,
        face_crop: np.ndarray | None,
        is_valid_sample: bool,
        face_landmarks: np.ndarray | None,
        status: str,
    ) -> np.ndarray:
        panel = np.full(
            (
                cls.PANEL_HEIGHT,
                frame_width,
                3,
            ),
            cls.BACKGROUND_COLOR,
            dtype=np.uint8,
        )

        cls._draw_face_crop(
            panel=panel,
            face_crop=face_crop,
            face_landmarks=face_landmarks,
            is_valid_sample=is_valid_sample,
        )

        cls._draw_information(
            panel=panel,
            face_crop=face_crop,
            is_valid_sample=is_valid_sample,
            status=status,
        )

        return panel

    @classmethod
    def _draw_face_crop(
        cls,
        panel: np.ndarray,
        face_crop: np.ndarray | None,
        face_landmarks: np.ndarray | None,
        is_valid_sample: bool,
    ) -> None:
        x1 = cls.PADDING
        y1 = cls.PADDING
        x2 = x1 + cls.CROP_SIZE
        y2 = y1 + cls.CROP_SIZE

        if face_crop is None or face_crop.size == 0:
            cv2.rectangle(
                panel,
                (x1, y1),
                (x2, y2),
                cls.EMPTY_COLOR,
                2,
            )

            cv2.putText(
                panel,
                "NO FACE",
                (x1 + 20, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                cls.EMPTY_COLOR,
                1,
            )
            return

        crop_preview, scale, x_offset, y_offset = (
            cls._resize_with_padding(
                image=face_crop,
                target_size=cls.CROP_SIZE,
            )
        )

        if face_landmarks is not None:
            cls._draw_landmarks(
                image=crop_preview,
                landmarks=face_landmarks,
                scale=scale,
                x_offset=x_offset,
                y_offset=y_offset,
            )

        panel[y1:y2, x1:x2] = crop_preview

        border_color = (
            cls.VALID_COLOR
            if is_valid_sample
            else cls.INVALID_COLOR
        )

        cv2.rectangle(
            panel,
            (x1, y1),
            (x2, y2),
            border_color,
            3,
        )

    @staticmethod
    def _draw_landmarks(
        image: np.ndarray,
        landmarks: np.ndarray,
        scale: float,
        x_offset: int,
        y_offset: int,
    ) -> None:
        points = np.asarray(
            landmarks,
            dtype=np.float32,
        ).reshape(-1, 2)

        colors = [
            (255, 0, 0),      # mắt 1
            (0, 255, 255),    # mắt 2
            (0, 0, 255),      # mũi
            (255, 0, 255),    # khóe miệng 1
            (0, 255, 0),      # khóe miệng 2
        ]

        for index, (landmark_x, landmark_y) in enumerate(points):
            draw_x = int(landmark_x * scale + x_offset)
            draw_y = int(landmark_y * scale + y_offset)

            color = colors[index % len(colors)]

            cv2.circle(
                image,
                (draw_x, draw_y),
                3,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

            cv2.putText(
                image,
                str(index),
                (draw_x + 4, draw_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def _resize_with_padding(
        cls,
        image: np.ndarray,
        target_size: int,
    ) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]

        scale = min(
            target_size / width,
            target_size / height,
        )

        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
        )

        canvas = np.zeros(
            (target_size, target_size, 3),
            dtype=np.uint8,
        )

        x_offset = (target_size - resized_width) // 2
        y_offset = (target_size - resized_height) // 2

        canvas[
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width,
        ] = resized

        return canvas, scale, x_offset, y_offset