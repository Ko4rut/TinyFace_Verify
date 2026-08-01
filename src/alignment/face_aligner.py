import cv2
import numpy as np

from src.alignment.models import FaceAlignmentResult


class FaceAligner:
    """
    Căn chỉnh khuôn mặt dựa trên hai landmark mắt.

    Thứ tự landmark:
        0: mắt trái ảnh
        1: mắt phải ảnh
        2: mũi
        3: khóe miệng trái ảnh
        4: khóe miệng phải ảnh
    """

    OUTPUT_SIZE = (112, 112)

    # Hai mắt đích nằm ngang tuyệt đối.
    REFERENCE_EYES = np.array(
        [
            [38.2946, 51.6],
            [73.5318, 51.6],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
    ) -> None:
        output_width, output_height = output_size

        if output_width <= 0 or output_height <= 0:
            raise ValueError(
                "output_size values must be greater than zero"
            )

        self.output_size = output_size

        self.reference_eyes = self._scale_reference_eyes(
            output_size=output_size,
        )

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> FaceAlignmentResult | None:
        """
        Căn chỉnh ảnh sao cho:

        - Hai mắt nằm ngang.
        - Khoảng cách hai mắt được chuẩn hóa.
        - Tâm hai mắt được đưa về vị trí chuẩn.
        """
        if image is None or image.size == 0:
            return None

        try:
            source_landmarks = self._normalize_landmarks(
                landmarks=landmarks,
            )
        except (TypeError, ValueError):
            return None

        source_eyes = source_landmarks[:2]

        # Alignment chỉ dùng hai mắt nên chỉ cần kiểm tra hai mắt.
        if not self._landmarks_are_inside_image(
            landmarks=source_eyes,
            image_shape=image.shape,
        ):
            return None

        transform_matrix = self._create_transform_matrix(
            source_eyes=source_eyes,
        )

        if transform_matrix is None:
            return None

        aligned_image = cv2.warpAffine(
            src=image,
            M=transform_matrix,
            dsize=self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Dùng cùng ma trận để chuyển toàn bộ 5 landmark.
        aligned_landmarks = cv2.transform(
            source_landmarks.reshape(1, 5, 2),
            transform_matrix,
        ).reshape(5, 2)

        return FaceAlignmentResult(
            image=aligned_image,
            landmarks=aligned_landmarks,
            transform_matrix=transform_matrix,
        )

    @staticmethod
    def calculate_eye_angle(
        left_eye: np.ndarray,
        right_eye: np.ndarray,
    ) -> float:
        """
        Tính góc của đường nối hai mắt so với phương ngang.

        Góc dương:
            mắt phải ảnh thấp hơn mắt trái ảnh.

        Góc âm:
            mắt phải ảnh cao hơn mắt trái ảnh.
        """
        left_eye = np.asarray(
            left_eye,
            dtype=np.float32,
        ).reshape(2)

        right_eye = np.asarray(
            right_eye,
            dtype=np.float32,
        ).reshape(2)

        eye_vector = right_eye - left_eye

        dx = float(eye_vector[0])
        dy = float(eye_vector[1])

        if np.hypot(dx, dy) <= 1e-6:
            raise ValueError(
                "The two eye landmarks must not overlap"
            )

        angle_radians = np.arctan2(dy, dx)

        return float(
            np.degrees(angle_radians)
        )

    def _create_transform_matrix(
        self,
        source_eyes: np.ndarray,
    ) -> np.ndarray | None:
        """
        Tạo ma trận similarity transform:

            M = [sR | t]

        Trong đó:
            R: phép xoay
            s: tỉ lệ
            t: phép tịnh tiến
        """
        source_left_eye = source_eyes[0]
        source_right_eye = source_eyes[1]

        target_left_eye = self.reference_eyes[0]
        target_right_eye = self.reference_eyes[1]

        source_vector = (
            source_right_eye - source_left_eye
        )

        target_vector = (
            target_right_eye - target_left_eye
        )

        source_distance = float(
            np.linalg.norm(source_vector)
        )

        target_distance = float(
            np.linalg.norm(target_vector)
        )

        if source_distance <= 1e-6:
            return None

        # Góc nghiêng hiện tại của hai mắt.
        source_angle_degrees = self.calculate_eye_angle(
            left_eye=source_left_eye,
            right_eye=source_right_eye,
        )

        # REFERENCE_EYES đang nằm ngang nên target_angle = 0°.
        # Ảnh nghiêng +10° thì cần xoay lại -10°.
        rotation_angle_degrees = -source_angle_degrees

        scale = target_distance / source_distance

        rotation_angle_radians = np.radians(
            rotation_angle_degrees
        )

        cos_value = (
            np.cos(rotation_angle_radians) * scale
        )

        sin_value = (
            np.sin(rotation_angle_radians) * scale
        )

        linear_transform = np.array(
            [
                [cos_value, -sin_value],
                [sin_value, cos_value],
            ],
            dtype=np.float32,
        )

        source_center = (
            source_left_eye + source_right_eye
        ) / 2.0

        target_center = (
            target_left_eye + target_right_eye
        ) / 2.0

        translation = (
            target_center
            - linear_transform @ source_center
        )

        transform_matrix = np.column_stack(
            [
                linear_transform,
                translation,
            ]
        ).astype(np.float32)

        if not np.isfinite(transform_matrix).all():
            return None

        return transform_matrix

    @classmethod
    def _scale_reference_eyes(
        cls,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        output_width, output_height = output_size

        coordinate_scale = np.array(
            [
                output_width / cls.OUTPUT_SIZE[0],
                output_height / cls.OUTPUT_SIZE[1],
            ],
            dtype=np.float32,
        )

        return (
            cls.REFERENCE_EYES.copy()
            * coordinate_scale
        )

    @staticmethod
    def _normalize_landmarks(
        landmarks: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if points.size != 10:
            raise ValueError(
                "landmarks must contain exactly five points"
            )

        points = points.reshape(5, 2).copy()

        if not np.isfinite(points).all():
            raise ValueError(
                "landmarks must contain finite values"
            )

        return points

    @staticmethod
    def _landmarks_are_inside_image(
        landmarks: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> bool:
        image_height, image_width = image_shape[:2]

        x_coordinates = landmarks[:, 0]
        y_coordinates = landmarks[:, 1]

        return bool(
            np.all(
                (x_coordinates >= 0)
                & (x_coordinates < image_width)
            )
            and np.all(
                (y_coordinates >= 0)
                & (y_coordinates < image_height)
            )
        )