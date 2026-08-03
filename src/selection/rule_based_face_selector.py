from math import hypot

from src.detection.models import FaceDetection
from src.selection.models import FaceSelectionResult, SelectionStatus


class RuleBasedFaceSelector:
    def __init__(
        self,
        area_weight: float = 0.6,
        center_weight: float = 0.4,
        ambiguity_margin: float = 0.05,
    ) -> None:
        """
        area_weight: The ratio of the face to frame
        center_weight: 
        """
        
        if area_weight < 0 or center_weight < 0:
            raise ValueError("Score weights must be non-negative")

        if area_weight + center_weight == 0:
            raise ValueError("At least one score weight must be positive")

        if not 0 <= ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be between 0 and 1")

        total_weight = area_weight + center_weight
        self.area_weight = area_weight / total_weight
        self.center_weight = center_weight / total_weight
        self.ambiguity_margin = ambiguity_margin

    def select(
        self,
        detections: list[FaceDetection],
        frame_shape: tuple[int, ...],
    ) -> FaceSelectionResult:
        if not detections:
            return FaceSelectionResult(status=SelectionStatus.NO_FACE)

        if len(frame_shape) < 2:
            raise ValueError("frame_shape must contain height and width")

        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("Frame dimensions must be positive")

        if len(detections) == 1:
            return FaceSelectionResult(
                status=SelectionStatus.SELECTED,
                face=detections[0],
            )

        largest_area = max(self._area(face) for face in detections)
        scored_faces = sorted(
            (
                (
                    self._score(
                        face,
                        frame_width,
                        frame_height,
                        largest_area,
                    ),
                    face,
                )
                for face in detections
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_face = scored_faces[0]
        second_score, _ = scored_faces[1]

        if best_score - second_score < self.ambiguity_margin:
            return FaceSelectionResult(status=SelectionStatus.AMBIGUOUS)

        return FaceSelectionResult(
            status=SelectionStatus.SELECTED,
            face=best_face,
        )

    def _score(
        self,
        face: FaceDetection,
        frame_width: int,
        frame_height: int,
        largest_area: int,
    ) -> float:
        x1, y1, x2, y2 = face.bbox
        area_score = (
            self._area(face) / largest_area
            if largest_area > 0
            else 0.0
        )

        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        distance = hypot(
            face_center_x - frame_center_x,
            face_center_y - frame_center_y,
        )
        max_distance = hypot(frame_center_x, frame_center_y)
        center_score = max(0.0, 1.0 - distance / max_distance)

        return (
            self.area_weight * area_score
            + self.center_weight * center_score
        )

    @staticmethod
    def _area(face: FaceDetection) -> int:
        x1, y1, x2, y2 = face.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
