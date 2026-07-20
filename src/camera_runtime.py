from collections import deque
import time

import cv2
import numpy as np

from src.camera_input import CameraInput
from src.preprocessing.detectors.yunet_detector import YuNetFaceDetector
from src.preprocessing.models import FaceDetection
from src.preprocessing.visualization.face_renderer import FaceRenderer

class CameraRuntime:
    def __init__(
        self,
        camera: CameraInput,
        face_detector: YuNetFaceDetector,
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
        self.face_detector = face_detector

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

    @staticmethod
    def get_detection_status(
        detections: list[FaceDetection],
    ) -> tuple[str, tuple[int, int, int]]:
        if not detections:
            return "No face detected", (0, 0, 255)

        if len(detections) > 1:
            return "Multiple faces detected", (0, 165, 255)

        return "Face detected", (0, 255, 0)
    
    def run(self) -> None:
        """Run the camera capture and face detection pipeline."""
        self.reset_session()

        try:
            for raw_frame in self.camera.face_from_camera():
                now = time.monotonic()

                # Flip trước để tạo ảnh hiển thị.
                preview = cv2.flip(raw_frame, 1)

                # Detect và vẽ trực tiếp theo tọa độ của preview.
                detections = self.face_detector.detect(preview)

                preview = FaceRenderer.draw(
                    preview,
                    detections,
                )

                if self.session_has_timed_out(now):
                    print("Session timeout. Resetting...")
                    self.reset_session()

                status, status_color = (
                    self.get_detection_status(detections)
                )

                if (
                    len(detections) == 1
                    and self.should_sample(now)
                ):
                    self.collect_frame(raw_frame, now)

                    print(
                        f"Collected: "
                        f"{len(self.selected_frames)}/"
                        f"{self.required_frames}"
                    )

                # 5. Hiển thị số frame đã thu.
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

                # 6. Hiển thị trạng thái detection.
                cv2.putText(
                    preview,
                    status,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )

                cv2.imshow("TinyFace Verify", preview)

                if self.is_session_complete():
                    frames = [
                        frame.copy()
                        for frame in self.selected_frames
                    ]

                    print("Session completed")

                    # Bước tiếp theo:
                    # result = self.verification_service.verify(frames)

                    self.reset_session()

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                if key == ord("r"):
                    self.reset_session()

                if (
                    cv2.getWindowProperty(
                        "TinyFace Verify",
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1
                ):
                    break

        finally:
            self.camera.close()
            cv2.destroyAllWindows()