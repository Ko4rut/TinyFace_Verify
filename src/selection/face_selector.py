from typing import Protocol

from src.detection.models import FaceDetection
from src.selection.models import FaceSelectionResult


class FaceSelector(Protocol):
    def select(
        self,
        detections: list[FaceDetection],
        frame_shape: tuple[int, ...],
    ) -> FaceSelectionResult:
        """Select the intended face, or report why none was selected."""
        ...
