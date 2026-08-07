import numpy as np


class EnrollmentSampleCollector:
    """
    Collects already-valid aligned face samples
    during one enrollment session.
    """

    def __init__(
        self,
        required_samples: int = 20,
        sample_interval_seconds: float = 0.3,
    ) -> None:
        self.required_samples = required_samples
        self.sample_interval_seconds = sample_interval_seconds

        self.samples: list[np.ndarray] = []
        self.last_sampled_at: float | None = None

    def try_add(
        self,
        aligned_face: np.ndarray,
        sampled_at: float,
    ) -> bool:
        """
        Adds a sample when enrollment is not complete
        and the sampling interval has passed.
        """
        if len(self.samples) >= self.required_samples:
            return False

        if (
            self.last_sampled_at is not None
            and sampled_at - self.last_sampled_at
            < self.sample_interval_seconds
        ):
            return False

        self.samples.append(aligned_face.copy())
        self.last_sampled_at = sampled_at

        return True

    @property
    def is_complete(self) -> bool:
        """Returns True when enough samples are collected."""
        return len(self.samples) >= self.required_samples

    def reset(self) -> None:
        """Starts a new enrollment collection session."""
        self.samples.clear()
        self.last_sampled_at = None