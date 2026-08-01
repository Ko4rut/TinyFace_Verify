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
        original_landmarks: np.ndarray | None,
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
            original_landmarks=original_landmarks,
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
        original_landmarks: np.ndarray | None,
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
            cls._draw_eye_comparison(
                image=crop_preview,
                aligned_landmarks=face_landmarks,
                original_landmarks=original_landmarks,
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
    def _draw_eye_alignment(
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

        if len(points) < 2:
            return

        left_eye = points[0]
        right_eye = points[1]

        left_eye_draw = (
            int(left_eye[0] * scale + x_offset),
            int(left_eye[1] * scale + y_offset),
        )

        right_eye_draw = (
            int(right_eye[0] * scale + x_offset),
            int(right_eye[1] * scale + y_offset),
        )

        # Vẽ hai mắt
        cv2.circle(
            image,
            left_eye_draw,
            4,
            (255, 0, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            right_eye_draw,
            4,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # Đường nối hai mắt
        cv2.line(
            image,
            left_eye_draw,
            right_eye_draw,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        dx = right_eye_draw[0] - left_eye_draw[0]
        dy = right_eye_draw[1] - left_eye_draw[1]

        if np.hypot(dx, dy) <= 1e-6:
            return

        angle = float(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        center_x = (
            left_eye_draw[0] + right_eye_draw[0]
        ) // 2

        center_y = (
            left_eye_draw[1] + right_eye_draw[1]
        ) // 2

        eye_distance = int(np.hypot(dx, dy))

        # Đường ngang tham chiếu
        cv2.line(
            image,
            (center_x - eye_distance // 2, center_y),
            (center_x + eye_distance // 2, center_y),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{angle:.1f} deg",
            (
                max(2, center_x - 30),
                max(12, center_y - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
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
            
    @staticmethod
    def _draw_eye_comparison(
        image: np.ndarray,
        aligned_landmarks: np.ndarray,
        original_landmarks: np.ndarray | None,
        scale: float,
        x_offset: int,
        y_offset: int,
    ) -> None:
        aligned_points = np.asarray(
            aligned_landmarks,
            dtype=np.float32,
        ).reshape(-1, 2)

        if len(aligned_points) < 2:
            return

        aligned_left_eye = aligned_points[0]
        aligned_right_eye = aligned_points[1]

        aligned_left_draw = (
            int(aligned_left_eye[0] * scale + x_offset),
            int(aligned_left_eye[1] * scale + y_offset),
        )

        aligned_right_draw = (
            int(aligned_right_eye[0] * scale + x_offset),
            int(aligned_right_eye[1] * scale + y_offset),
        )

        # Hai điểm mắt sau align
        cv2.circle(
            image,
            aligned_left_draw,
            3,
            (255, 0, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            aligned_right_draw,
            3,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # Đường mắt sau align — màu xanh lá
        cv2.line(
            image,
            aligned_left_draw,
            aligned_right_draw,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        aligned_dx = (
            aligned_right_draw[0] - aligned_left_draw[0]
        )
        aligned_dy = (
            aligned_right_draw[1] - aligned_left_draw[1]
        )

        aligned_angle = float(
            np.degrees(
                np.arctan2(aligned_dy, aligned_dx)
            )
        )

        center_x = (
            aligned_left_draw[0] + aligned_right_draw[0]
        ) // 2

        center_y = (
            aligned_left_draw[1] + aligned_right_draw[1]
        ) // 2

        aligned_distance = float(
            np.hypot(aligned_dx, aligned_dy)
        )

        # Vẽ góc trước align để so sánh
        if original_landmarks is not None:
            original_points = np.asarray(
                original_landmarks,
                dtype=np.float32,
            ).reshape(-1, 2)

            if len(original_points) >= 2:
                original_vector = (
                    original_points[1] - original_points[0]
                )

                original_distance = float(
                    np.linalg.norm(original_vector)
                )

                if original_distance > 1e-6:
                    original_unit_vector = (
                        original_vector / original_distance
                    )

                    # Dùng cùng độ dài với đường aligned để dễ so sánh
                    half_vector = (
                        original_unit_vector
                        * aligned_distance
                        / 2.0
                    )

                    original_start = (
                        int(center_x - half_vector[0]),
                        int(center_y - half_vector[1]),
                    )

                    original_end = (
                        int(center_x + half_vector[0]),
                        int(center_y + half_vector[1]),
                    )

                    # Đường mắt trước align — màu đỏ
                    cv2.line(
                        image,
                        original_start,
                        original_end,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    original_angle = float(
                        np.degrees(
                            np.arctan2(
                                original_vector[1],
                                original_vector[0],
                            )
                        )
                    )

                    cv2.putText(
                        image,
                        f"Before: {original_angle:.1f}",
                        (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        cv2.putText(
            image,
            f"After: {aligned_angle:.1f}",
            (3, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
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
    
    @classmethod
    def _draw_information(
        cls,
        panel: np.ndarray,
        face_crop: np.ndarray | None,
        is_valid_sample: bool,
        status: str,
    ) -> None:
        text_x = cls.PADDING + cls.CROP_SIZE + 25

        has_face_crop = (
            face_crop is not None
            and face_crop.size > 0
        )

        if not has_face_crop:
            sample_text = "WAITING FOR FACE"
            sample_color = cls.EMPTY_COLOR
        elif is_valid_sample:
            sample_text = "READY TO COLLECT"
            sample_color = cls.VALID_COLOR
        else:
            sample_text = "NOT READY"
            sample_color = cls.INVALID_COLOR

        cv2.putText(
            panel,
            sample_text,
            (text_x, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            sample_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            status,
            (text_x, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cls.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )