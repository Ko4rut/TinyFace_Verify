from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceAlignmentResult:
    """Result of aligning one cropped face to the reference template."""

    image: np.ndarray
    landmarks: np.ndarray
    transform_matrix: np.ndarray


class FaceAligner:
    """
    Align a cropped face using a 2D similarity transformation.

    Landmark order must follow image coordinates from left to right:
        image-left eye, image-right eye, nose,
        image-left mouth corner, image-right mouth corner.

    The input landmarks must use the coordinate system of the input image.
    """

    OUTPUT_SIZE = (112, 112)

    # ArcFace five-point template for a 112 x 112 output image.
    REFERENCE_LANDMARKS = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        reference_landmarks: np.ndarray | None = None,
    ) -> None:
        output_width, output_height = output_size

        if output_width <= 0 or output_height <= 0:
            raise ValueError("output_size values must be greater than zero")

        self.output_size = (output_width, output_height)

        if reference_landmarks is None:
            self.reference_landmarks = self._scaled_reference_landmarks(
                output_size=self.output_size,
            )
        else:
            self.reference_landmarks = self._normalize_landmarks(
                reference_landmarks,
                name="reference_landmarks",
            )

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> FaceAlignmentResult | None:
        """
        Align an image from its five source landmarks.

        Returns None when the image or landmarks cannot produce a valid
        similarity transformation.
        """
        if image is None or image.size == 0:
            return None

        try:
            source_landmarks = self._normalize_landmarks(
                landmarks,
                name="landmarks",
            )
        except (TypeError, ValueError):
            return None

        if not self._landmarks_are_inside_image(
            landmarks=source_landmarks,
            image_shape=image.shape,
        ):
            return None

        transform_matrix, _ = cv2.estimateAffinePartial2D(
            source_landmarks,
            self.reference_landmarks,
            method=cv2.LMEDS,
        )

        if transform_matrix is None:
            return None

        transform_matrix = np.asarray(
            transform_matrix,
            dtype=np.float32,
        )

        if (
            transform_matrix.shape != (2, 3)
            or not np.isfinite(transform_matrix).all()
        ):
            return None

        aligned_image = cv2.warpAffine(
            src=image,
            M=transform_matrix,
            dsize=self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        aligned_landmarks = cv2.transform(
            source_landmarks.reshape(1, 5, 2),
            transform_matrix,
        ).reshape(5, 2)

        return FaceAlignmentResult(
            image=aligned_image,
            landmarks=aligned_landmarks,
            transform_matrix=transform_matrix,
        )

    @classmethod
    def _scaled_reference_landmarks(
        cls,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        output_width, output_height = output_size
        scale = np.array(
            [
                output_width / cls.OUTPUT_SIZE[0],
                output_height / cls.OUTPUT_SIZE[1],
            ],
            dtype=np.float32,
        )
        return cls.REFERENCE_LANDMARKS.copy() * scale

    @staticmethod
    def _normalize_landmarks(
        landmarks: np.ndarray,
        name: str,
    ) -> np.ndarray:
        points = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if points.size != 10:
            raise ValueError(f"{name} must contain exactly five (x, y) points")

        points = points.reshape(5, 2).copy()

        if not np.isfinite(points).all():
            raise ValueError(f"{name} must contain only finite values")

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
            np.all((0 <= x_coordinates) & (x_coordinates < image_width))
            and np.all((0 <= y_coordinates) & (y_coordinates < image_height))
        )