import time

import cv2

from src.camera.camera_input import CameraInput
from src.capture.frame_session import FrameSession
from src.detection.yunet_detector import YuNetFaceDetector
from src.draw.camera_renderer_services import CameraRenderer
from src.selection.face_selector import FaceSelector
from src.selection.models import (
    FaceSelectionResult,
    SelectionStatus,
)
from src.utils.face_helper import crop_face
from src.validation.face_sample_validator import FaceSampleValidator
from src.draw.models import CameraRenderState
from src.alignment.face_aligner import FaceAligner

class CameraRuntime:
    WINDOW_NAME = "TinyFace Verify"

    def __init__(
        self,
        camera: CameraInput,
        face_detector: YuNetFaceDetector,
        session: FrameSession,
        face_selector: FaceSelector,
        face_validator: FaceSampleValidator,
        face_aligner: FaceAligner,
    ) -> None:
        self.camera = camera
        self.face_detector = face_detector
        self.session = session
        self.face_selector = face_selector
        self.face_validator = face_validator
        self.face_aligner = face_aligner
    @staticmethod
    def get_selection_status(
        selection: FaceSelectionResult,
        is_valid_sample: bool,
    ) -> tuple[str, tuple[int, int, int]]:
        if selection.status is SelectionStatus.NO_FACE:
            return "No face detected", (0, 0, 255)

        if selection.status is SelectionStatus.AMBIGUOUS:
            return "Cannot determine target face", (0, 165, 255)

        if not is_valid_sample:
            return "Face selected, but sample is invalid", (0, 165, 255)

        return "Target face is ready", (0, 255, 0)

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
                if raw_frame is None or raw_frame.size == 0:
                    continue

                now = time.monotonic()

                # Dùng cùng một frame cho detect, crop, validate và hiển thị.
                preview = cv2.flip(raw_frame, 1)

                if self.session.has_timed_out(now):
                    print("Session timeout. Resetting...")
                    self.session.reset()

                # 1. Phát hiện tất cả khuôn mặt
                detections = self.face_detector.detect(preview)

                # 2. Chọn khuôn mặt mục tiêu
                selection = self.face_selector.select(
                    detections=detections,
                    frame_shape=preview.shape,
                )

                selected_face = None

                if selection.status is SelectionStatus.SELECTED:
                    selected_face = selection.face

                # 3. Crop ảnh và chuyển landmark về tọa độ crop
                cropped_face = None
                
                if selected_face is not None:
                    cropped_face = crop_face(
                        frame=preview,
                        face=selected_face,
                    )

                # 4. Kiểm tra chất lượng mẫu
                is_valid_sample = False

                if selected_face is not None and cropped_face is not None:
                    is_valid_sample = self.face_validator.validate(
                        frame=preview,
                        face=selected_face,
                    )

                # 5. Tạo trạng thái hiển thị
                status, status_color = self.get_selection_status(
                    selection=selection,
                    is_valid_sample=is_valid_sample,
                )

                # 6. Thu thập mẫu hợp lệ
                if (
                    is_valid_sample
                    and cropped_face is not None
                    and self.session.should_sample(now)
                ):
                    self.session.add(
                        preview.copy(),
                        now,
                    )

                    print(
                        f"Collected: {self.session.collected_count}/"
                        f"{self.session.required_frames}"
                    )

                # 7. Tách ảnh và landmark để renderer sử dụng
                face_crop_image = (
                    cropped_face.image
                    if cropped_face is not None
                    else None
                )

                face_crop_landmarks = (
                    cropped_face.landmarks
                    if cropped_face is not None
                    else None
                )

                
                render_state = CameraRenderState(
                    detections=detections,
                    selected_face=selected_face,
                    face_crop=face_crop_image,
                    crop_landmarks=face_crop_landmarks,
                    is_valid_sample=is_valid_sample,
                    collected_count=self.session.collected_count,
                    required_frames=self.session.required_frames,
                    status=status,
                    status_color=status_color
                )
                # 8. Render camera và panel
                rendered_frame = CameraRenderer.render(
                    frame=preview,
                    state = render_state
                )

                cv2.imshow(
                    self.WINDOW_NAME,
                    rendered_frame,
                )

                # 9. Xử lý session hoàn tất
                if self.session.is_complete:
                    print("Session completed")

                    frames = self.session.get_frames()

                    # Sau này gọi:
                    # result = self.verification_service.verify(frames)

                    self.session.reset()

                # 10. Xử lý bàn phím
                key = cv2.waitKey(1) & 0xFF

                if self.should_stop(key):
                    break

                if key == ord("r"):
                    print("Session manually reset")
                    self.session.reset()

        finally:
            self.camera.close()
            cv2.destroyAllWindows()