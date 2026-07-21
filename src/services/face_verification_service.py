from dataclasses import dataclass
from enum import Enum

import numpy as np


class VerificationStatus(Enum):
    NOT_IMPLEMENTED = "not_implemented"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    message: str


class FaceVerificationService:
    def verify(
        self,
        frames: list[np.ndarray],
    ) -> VerificationResult:
        if not frames:
            raise ValueError("Frames must not be empty")

        return VerificationResult(
            status=VerificationStatus.NOT_IMPLEMENTED,
            message=(
                f"Captured {len(frames)} valid frames. "
                "Face verification is not implemented yet."
            ),
        )