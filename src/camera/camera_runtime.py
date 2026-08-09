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
from src.enrollment.enrollment_sample_collector import (
    EnrollmentSampleCollector,
)
from src.enrollment.enrollment_service import (
    EnrollmentService,
)
from src.verification.face_verification_service import (
    FaceVerificationService,
)



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
        enrollment_service: EnrollmentService | None,
        verification_service: FaceVerificationService | None,
        enrollment_collector: EnrollmentSampleCollector | None = None,
    ) -> None:
        self.camera = camera
        self.face_detector = face_detector
        self.session = session
        self.face_selector = face_selector
        self.face_validator = face_validator
        self.face_aligner = face_aligner

        self.enrollment_service = enrollment_service
        self.verification_service = verification_service
        self.enrollment_collector = enrollment_collector
        self.verification_result = None
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
        if self.is_enrollment_mode:
            self.enrollment_collector.reset()
        else:
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
        preview = cv2.flip(raw_frame, 1)

        # 1. Kiểm tra timeout
        if self.session.has_timed_out(now):
            print("Session timeout. Resetting...")
            self.session.reset()

        # 2. Detect khuôn mặt
        detections = self.face_detector.detect(preview)

        # 3. Chọn khuôn mặt mục tiêu
        selection = self.face_selector.select(
            detections=detections,
            frame_shape=preview.shape,
        )

        selected_face = (
            selection.face
            if selection.status is SelectionStatus.SELECTED
            else None
        )

        # 4. Validate
        is_valid_sample = False

        if selected_face is not None:
            is_valid_sample = self.face_validator.validate(
                frame=preview,
                face=selected_face,
            )

        # 5. Align trực tiếp từ frame
        alignment_result = None

        if is_valid_sample and selected_face is not None:
            alignment_result = self.face_aligner.align(
                image=preview,
                landmarks=selected_face.landmarks,
            )

        # 6. Kiểm tra mẫu đã sẵn sàng
        is_sample_ready = (
            is_valid_sample
            and alignment_result is not None
        )

        # 7. Thu ảnh aligned
        if is_sample_ready:
            aligned_face = alignment_result.image

            if self.is_enrollment_mode:
                was_added = self.enrollment_collector.try_add(
                    aligned_face=aligned_face,
                    sampled_at=now,
                )

                if was_added:
                    print(
                        f"Enrollment collected: "
                        f"{len(self.enrollment_collector.samples)}/"
                        f"{self.enrollment_collector.required_samples}"
                    )

            elif self.session.should_sample(now):
                self.session.add(
                    aligned_face.copy(),
                    now,
                )
                
                print(
                    f"Verification collected: "
                    f"{self.session.collected_count}/"
                    f"{self.session.required_frames}"
                )

        # 8. Tạo status
        status, status_color = self.get_selection_status(
            selection=selection,
            is_valid_sample=is_sample_ready,
        )

        render_state = CameraRenderState(
            detections=detections,
            selected_face=selected_face,

            face_crop=(
                alignment_result.image
                if alignment_result is not None
                else None
            ),

            # Landmark sau align
            crop_landmarks=(
                alignment_result.landmarks
                if alignment_result is not None
                else None
            ),

            # Landmark trước align
            original_landmarks=(
                selected_face.landmarks
                if selected_face is not None
                else None
            ),

            is_valid_sample=is_sample_ready,
            collected_count=self.session.collected_count,
            required_frames=self.session.required_frames,
            status=status,
            status_color=status_color,
            verification_result=self.verification_result
        )

        # 10. Render
        rendered_frame = CameraRenderer.render(
            frame=preview,
            state=render_state,
        )

        cv2.imshow(
            self.WINDOW_NAME,
            rendered_frame,
        )

        # 11. Handle completed collection
        if self.is_enrollment_mode:
            if self.enrollment_collector.is_complete:
                self._handle_completed_enrollment()
                return True

        elif self.session.is_complete:
            self._handle_completed_verification()

        # 12. Xử lý bàn phím
        key = cv2.waitKey(1) & 0xFF

        if self.should_stop(key):
            return True

        if key == ord("r"):
            print("Session manually reset")
            self.session.reset()

        return False

    def _handle_completed_enrollment(self) -> None:
        print("Enrollment completed")

        if self.enrollment_service is None:
            raise RuntimeError(
                "EnrollmentService is required in enrollment mode."
            )

        self.enrollment_service.enroll(
            aligned_faces=self.enrollment_collector.samples,
        )
        
    def _handle_completed_verification(self) -> None:
        print("Verification samples collected")

        if self.verification_service is None:
            raise RuntimeError(
                "FaceVerificationService is required "
                "in verification mode."
            )

        is_match, mean_error, frame_errors = (
            self.verification_service.verify(
                aligned_faces=self.session.get_frames(),
            )
        )

        self.verification_result = is_match

        print(f"Mean reconstruction error: {mean_error:.4f}")
        print(f"Frame errors: {frame_errors}")

        if is_match:
            print("Verification successful: owner matched.")
        else:
            print("Verification failed: owner not matched.")

        # Bắt đầu một lượt thu 5 frame mới.
        self.session.reset()
    
    @property
    def is_enrollment_mode(self) -> bool:
        return self.enrollment_collector is not None