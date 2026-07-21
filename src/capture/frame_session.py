from collections import deque

import numpy as np


class FrameSession:
    def __init__(
        self,
        required_frames: int = 5,
        sample_interval: float = 0.15,
        timeout: float = 2.0,
    ) -> None:
        self.required_frames = required_frames
        self.sample_interval = sample_interval
        self.timeout = timeout

        self.frames: deque[np.ndarray] = deque(
            maxlen=required_frames
        )
        self.started_at: float | None = None
        self.last_sampled_at: float | None = None

    @property
    def collected_count(self) -> int:
        return len(self.frames)

    @property
    def is_complete(self) -> bool:
        return len(self.frames) == self.required_frames

    def has_timed_out(self, now: float) -> bool:
        if self.started_at is None:
            return False

        return now - self.started_at >= self.timeout

    def should_sample(self, now: float) -> bool:
        if self.last_sampled_at is None:
            return True

        return now - self.last_sampled_at >= self.sample_interval

    def add(self, frame: np.ndarray, now: float) -> None:
        if self.started_at is None:
            self.started_at = now

        self.frames.append(frame.copy())
        self.last_sampled_at = now

    def get_frames(self) -> list[np.ndarray]:
        return [frame.copy() for frame in self.frames]

    def reset(self) -> None:
        self.frames.clear()
        self.started_at = None
        self.last_sampled_at = None