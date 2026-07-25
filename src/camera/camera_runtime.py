import time

import cv2

from src.camera.camera_input import CameraInput
from src.capture.frame_session import FrameSession
from src.detection.models import FaceDetection
from src.detection.yunet_detector import YuNetFaceDetector
from src.draw.camera_renderer import CameraRenderer
from src.selection.face_selector import FaceSelector
from src.selection.models import FaceSelectionResult, SelectionStatus
from src.validation.face_sample_validator import FaceSampleValidator
from src.utils.face_helper import crop_face

class CameraRuntime:
    WINDOW_NAME = "TinyFace Verify"

    def __init__(
        self,
        camera: CameraInput,
        face_detector: YuNetFaceDetector,
        session: FrameSession,
        face_selector: FaceSelector,
        face_validator: FaceSampleValidator
    ) -> None:
        self.camera = camera
        self.face_detector = face_detector
        self.session = session
        self.face_selector = face_selector
        self.face_validator = face_validator

    @staticmethod
    def get_selection_status(
        selection: FaceSelectionResult,
    ) -> tuple[str, tuple[int, int, int]]:
        if selection.status is SelectionStatus.NO_FACE:
            return "No face detected", (0, 0, 255)

        if selection.status is SelectionStatus.AMBIGUOUS:
            return "Cannot determine target face", (0, 165, 255)

        return "Target face selected", (0, 255, 0)
    
    def should_stop(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return True

        return (
            cv2.getWindowProperty(
                self.WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            )
            < 1
        )

    def run(self) -> None:
        self.session.reset()

        try:
            for raw_frame in self.camera.face_from_camera():
                now = time.monotonic()

                preview = cv2.flip(raw_frame, 1)
                
                if self.session.has_timed_out(now):
                    print("Session timeout. Resetting...")
                    self.session.reset()
                    
                detections = self.face_detector.detect(preview)


                selection = self.face_selector.select(
                                    detections=detections,
                                    frame_shape=preview.shape,
                                )
                selected_face = (
                                    selection.face
                                    if selection.status is SelectionStatus.SELECTED
                                    else None
                                )
                face_crop = (
                    crop_face(preview, selected_face)
                    if selected_face is not None
                    else None
                )

                is_valid_sample = (
                    selected_face is not None
                    and self.face_validator.validate(
                        frame=preview,
                        face=selected_face,
                    )
                )
                status, status_color = self.get_selection_status(
                    selection
                )
                is_valid_sample = (
                    selected_face is not None
                    and self.face_validator.validate(
                        frame=preview,
                        face=selected_face,
                    )
                )
                
                if is_valid_sample and self.session.should_sample(now):
                    self.session.add(preview, now)
                    
                    # print(
                    #     f"Collected: {self.session.collected_count}/"
                    #     f"{self.session.required_frames}"
                    # )

                preview = CameraRenderer.render(
                frame=preview,
                detections=detections,
                selected_face=selected_face,
                face_crop=face_crop,
                is_valid_sample=is_valid_sample,
                collected_count=self.session.collected_count,
                required_frames=self.session.required_frames,
                status=status,
                status_color=status_color,
            )
                cv2.imshow(self.WINDOW_NAME, preview)

                if self.session.is_complete:
                    # print("Session completed")

                    frames = self.session.get_frames()
                    
                    # result = self.verification_service.verify(frames)

                    self.session.reset()

                key = cv2.waitKey(1) & 0xFF

                if self.should_stop(key):
                    break

                if key == ord("r"):
                    self.session.reset()

        finally:
            self.camera.close()
            cv2.destroyAllWindows()