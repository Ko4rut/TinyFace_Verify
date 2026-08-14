"""Value objects produced by offline verification evaluation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    """False-accept and false-reject rates at one error threshold."""

    threshold: float
    false_accept_rate: float
    false_reject_rate: float

    @property
    def balanced_error_rate(self) -> float:
        """Return the mean of the two error rates."""
        return (self.false_accept_rate + self.false_reject_rate) / 2.0


@dataclass(frozen=True)
class VerificationEvaluation:
    """Scores and threshold metrics for one enrolled identity."""

    genuine_errors: np.ndarray
    impostor_errors: np.ndarray
    threshold_metrics: tuple[ThresholdMetrics, ...]
    recommended_threshold: ThresholdMetrics

    @property
    def genuine_sample_count(self) -> int:
        return int(self.genuine_errors.shape[0])

    @property
    def impostor_sample_count(self) -> int:
        return int(self.impostor_errors.shape[0])
