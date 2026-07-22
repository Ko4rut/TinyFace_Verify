from dataclasses import dataclass
from enum import Enum, auto

from src.detection.models import FaceDetection


class SelectionStatus(Enum):
    NO_FACE = auto()
    SELECTED = auto()
    AMBIGUOUS = auto()


@dataclass(frozen=True)
class FaceSelectionResult:
    status: SelectionStatus
    face: FaceDetection | None = None

    def __post_init__(self) -> None:
        if self.status is SelectionStatus.SELECTED and self.face is None:
            raise ValueError("SELECTED result must contain a face")

        if self.status is not SelectionStatus.SELECTED and self.face is not None:
            raise ValueError(
                "Only a SELECTED result may contain a face"
            )
