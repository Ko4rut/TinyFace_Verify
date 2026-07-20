import cv2
import numpy as np

from src.preprocessing.detectors.yunet_mapper import (
    map_yunet_detection,
)
from src.preprocessing.models import FaceDetection


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[FaceDetection]:
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))

        _, raw_detections = self.detector.detect(frame)

        if raw_detections is None:
            return []

        return [
            map_yunet_detection(raw_detection)
            for raw_detection in raw_detections
        ]