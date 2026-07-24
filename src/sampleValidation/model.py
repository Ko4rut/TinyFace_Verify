from dataclasses import dataclass
from enum import Enum, auto


class FaceQualityStatus(Enum):
    AVAILABLE = auto()
    TOO_SMALL = auto()
    TOO_CLOSE_TO_EDGE = auto()
    LOW_CONFIDENCE = auto()
    TOO_BLURRY = auto()
    BAD_POSE = auto()


@dataclass(frozen=True)
class FaceQualityResult:
    status: FaceQualityStatus

    @property
    def is_available(self) -> bool:
        return self.status == FaceQualityStatus.AVAILABLE