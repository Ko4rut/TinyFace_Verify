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

                should_stop = self._process_frame(raw_frame)

                if should_stop:
                    break
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
            
    def _process_frame(self, raw_frame) -> bool:
        """
        Xử lý một frame camera.

        Returns:
            True: dừng CameraRuntime.
            False: tiếp tục đọc frame tiếp theo.
        """
        now = time.monotonic()

        # Dùng cùng một frame cho detect, validate, crop và hiển thị.
        preview = cv2.flip(raw_frame, 1)

        # 1. Kiểm tra timeout của phiên thu mẫu
        if self.session.has_timed_out(now):
            print("Session timeout. Resetting...")
            self.session.reset()

        # 2. Phát hiện tất cả khuôn mặt
        detections = self.face_detector.detect(preview)

        # 3. Chọn khuôn mặt mục tiêu
        selection = self.face_selector.select(
            detections=detections,
            frame_shape=preview.shape,
        )

        selected_face = None

        if selection.status is SelectionStatus.SELECTED:
            selected_face = selection.face

        # 4. Khởi tạo kết quả xử lý khuôn mặt
        cropped_face = None
        alignment_result = None
        is_valid_sample = False

        # 5. Validate khuôn mặt được chọn
        if selected_face is not None:
            is_valid_sample = self.face_validator.validate(
                frame=preview,
                face=selected_face,
            )

        # 6. Crop để hiển thị và chuyển landmark sang tọa độ crop
        if selected_face is not None:
            cropped_face = crop_face(
                frame=preview,
                face=selected_face,
            )

        # 7. Chỉ alignment nếu mẫu đã qua validation
        if is_valid_sample and cropped_face is not None:
            alignment_result = self.face_aligner.align(
                image=cropped_face.image,
                landmarks=cropped_face.landmarks,
            )

        # Mẫu chỉ thực sự sẵn sàng khi đã:
        # selected → validated → cropped → aligned
        is_sample_ready = (
            is_valid_sample
            and cropped_face is not None
            and alignment_result is not None
        )

        # 8. Thu ảnh mặt đã alignment
        if (
            is_sample_ready
            and self.session.should_sample(now)
        ):
            self.session.add(
                alignment_result.image.copy(),
                now,
            )

            print(
                f"Collected: {self.session.collected_count}/"
                f"{self.session.required_frames}"
            )

        # 9. Tạo trạng thái thông báo
        status, status_color = self.get_selection_status(
            selection=selection,
            is_valid_sample=is_sample_ready,
        )

        # 10. Chuẩn bị dữ liệu cho renderer
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
        
        # 10. Chuẩn bị dữ liệu cho renderer
        if alignment_result is not None:
            display_face_image = alignment_result.image
            display_face_landmarks = alignment_result.landmarks
        elif cropped_face is not None:
            display_face_image = cropped_face.image
            display_face_landmarks = cropped_face.landmarks
        else:
            display_face_image = None
            display_face_landmarks = None

        render_state = CameraRenderState(
            detections=detections,
            selected_face=selected_face,
            face_crop=display_face_image,
            crop_landmarks=display_face_landmarks,
            is_valid_sample=is_sample_ready,
            collected_count=self.session.collected_count,
            required_frames=self.session.required_frames,
            status=status,
            status_color=status_color,
        )

        # 11. Render camera và panel khuôn mặt
        rendered_frame = CameraRenderer.render(
            frame=preview,
            state=render_state,
        )

        cv2.imshow(
            self.WINDOW_NAME,
            rendered_frame,
        )

        # 12. Xử lý khi đã thu đủ số lượng mẫu
        if self.session.is_complete:
            self._handle_completed_session()

        # 13. Xử lý bàn phím
        key = cv2.waitKey(1) & 0xFF

        if self.should_stop(key):
            return True

        if key == ord("r"):
            print("Session manually reset")
            self.session.reset()

        return False

    def _handle_completed_session(self) -> None:
        print("Session completed")

        aligned_faces = self.session.get_frames()

        # Sau này:
        # result = self.verification_service.verify(aligned_faces)
        # print(result)

        self.session.reset()