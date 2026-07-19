from collections import deque
import time

import cv2
import numpy as np

from src.camera_input import CameraInput


class CameraRuntime:
    def __init__(
        self,
        camera: CameraInput,
        required_frames: int = 5,
        sample_interval: float = 0.15,
        session_timeout: float = 2.0,
    ):
        self.camera = camera
        self.required_frames = required_frames  # Number of frames to collect in a session
        self.sample_interval = sample_interval  # Time interval between collecting two frames 
        self.session_timeout = session_timeout  # Maximum time allowed for a session before timeout

        self.selected_frames = deque(maxlen=required_frames) # Queue to store the collected frames
        self.session_started_at: float | None = None    # Timestamp of the first collected frame 
        self.last_sampled_at: float | None = None   # Timestamp of the most recently collected frame

    def reset_session(self) -> None:
        """Clear the session data"""
        self.selected_frames.clear()
        self.session_started_at = None
        self.last_sampled_at = None

    def session_has_timed_out(self, now: float) -> bool:
        """Check session is timeout?"""
        if self.session_started_at is None:
            return False

        return now - self.session_started_at >= self.session_timeout

    def should_sample(self, now: float) -> bool:
        """Decide can take the frame"""
        if self.last_sampled_at is None:
            return True

        return now - self.last_sampled_at >= self.sample_interval

    def collect_frame(
        self,
        frame: np.ndarray,
        now: float,
    ) -> None:
        """Collect frame"""
        if self.session_started_at is None:
            self.session_started_at = now

        self.selected_frames.append(frame.copy())
        self.last_sampled_at = now

    def is_session_complete(self) -> bool:
        """Check the session completed?"""
        return len(self.selected_frames) == self.required_frames

    def run(self) -> None:
        """Run the pipeline"""
        self.reset_session()

        try:
            for raw_frame in self.camera.face_from_camera():
                now = time.monotonic()

                preview = cv2.flip(raw_frame, 1)

                if self.session_has_timed_out(now):
                    print("Session timeout. Resetting...")
                    self.reset_session()

                if self.should_sample(now):
                    self.collect_frame(raw_frame, now)

                    print(
                        f"Collected: "
                        f"{len(self.selected_frames)}/"
                        f"{self.required_frames}"
                    )

                cv2.putText(
                    preview,
                    (
                        f"Collected: {len(self.selected_frames)}"
                        f"/{self.required_frames}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("TinyFace Verify", preview)

                if self.is_session_complete():
                    print("Session completed")
                    # sent frame to preproccess and verify after
                    self.reset_session()

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                if cv2.getWindowProperty(
                    "TinyFace Verify",
                    cv2.WND_PROP_VISIBLE,
                ) < 1:
                    break
                
                if key == ord("r"):
                    self.reset_session()
                    

        finally:
            self.camera.close()
            cv2.destroyAllWindows()