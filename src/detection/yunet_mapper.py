import numpy as np

from src.detection.models import FaceDetection


def map_yunet_detection(
    raw_detection: np.ndarray,
) -> FaceDetection:
    if raw_detection.shape[0] < 15:
        raise ValueError(
            "YuNet detection must contain at least 15 values"
        )

    x, y, width, height = raw_detection[:4].astype(int)

    landmarks = (
        raw_detection[4:14]
        .reshape(5, 2)
        .astype(np.float32)
    )

    return FaceDetection(
        bbox=(
            x,
            y,
            x + width,
            y + height,
        ),
        confidence=float(raw_detection[14]),
        landmarks=landmarks,
    )