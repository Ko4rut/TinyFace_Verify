"""Coordinate camera enrollment and face-verification sessions."""

import time

import cv2
import numpy as np

from src.alignment.face_aligner import FaceAligner
from src.camera.camera_input import CameraInput
from src.camera.models import FrameAnalysis
from src.capture.frame_session import FrameSession
from src.detection.yunet_detector import YuNetFaceDetector
from src.draw.camera_renderer_services import CameraRenderer
from src.camera.camera_render_state_builder import CameraRenderStateBuilder
from src.enrollment.enrollment_sample_collector import EnrollmentSampleCollector
from src.services.enrollment_service import EnrollmentService
from src.selection.face_selector import FaceSelector
from src.selection.models import SelectionStatus
from src.validation.face_sample_validator import FaceSampleValidator
from src.services.face_verification_service import FaceVerificationService


class CameraRuntimeService:
    """Run the live camera loop and coordinate the face-processing pipeline."""

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
        """Store the camera dependencies and select the operating mode.

        Providing ``enrollment_collector`` enables enrollment mode.
        Leaving it as ``None`` enables verification mode.
        """
        self.camera = camera
        self.face_detector = face_detector
        self.session = session
        self.face_selector = face_selector
        self.face_validator = face_validator
        self.face_aligner = face_aligner
        self.enrollment_service = enrollment_service
        self.verification_service = verification_service
        self.enrollment_collector = enrollment_collector
        self.verification_result: bool | None = None

    def run(self) -> None:
        """Read camera frames until the user exits or enrollment finishes."""
        self.session.reset()

        if self.is_enrollment_mode:
            self.enrollment_collector.reset()

        try:
            for raw_frame in self.camera.face_from_camera():
                if raw_frame is None or raw_frame.size == 0:
                    continue

                if self._process_frame(raw_frame):
                    break
        finally:
            self.camera.close()
            cv2.destroyAllWindows()

    def _process_frame(self, raw_frame: np.ndarray) -> bool:
        """Analyze, collect, render, and handle one camera frame."""
        now = time.monotonic()
        preview = cv2.flip(raw_frame, 1)

        self._reset_timed_out_verification(now)
        analysis = self._analyze_frame(preview)

        if analysis.is_sample_ready:
            alignment_result = analysis.alignment_result
            if alignment_result is not None:
                self._collect_aligned_face(
                    aligned_face=alignment_result.image,
                    sampled_at=now,
                )

        self._render_frame(frame=preview, analysis=analysis)

        if self._handle_completed_collection():
            return True

        return self._handle_keyboard()

    def _analyze_frame(self, preview: np.ndarray) -> FrameAnalysis:
        """Detect, select, validate, and align one candidate face."""
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

        if selected_face is None:
            return FrameAnalysis(
                detections=detections,
                selection=selection,
                selected_face=None,
                alignment_result=None,
                is_sample_ready=False,
            )

        is_valid = self.face_validator.validate(
            frame=preview,
            face=selected_face,
        )

        if not is_valid:
            return FrameAnalysis(
                detections=detections,
                selection=selection,
                selected_face=selected_face,
                alignment_result=None,
                is_sample_ready=False,
            )

        alignment_result = self.face_aligner.align(
            image=preview,
            landmarks=selected_face.landmarks,
        )

        return FrameAnalysis(
            detections=detections,
            selection=selection,
            selected_face=selected_face,
            alignment_result=alignment_result,
            is_sample_ready=alignment_result is not None,
        )

    def _collect_aligned_face(
        self,
        aligned_face: np.ndarray,
        sampled_at: float,
    ) -> None:
        """Add one valid aligned face to the active collection session."""
        if self.is_enrollment_mode:
            was_added = self.enrollment_collector.try_add(
                aligned_face=aligned_face,
                sampled_at=sampled_at,
            )

            if was_added:
                print(
                    "Enrollment collected: "
                    f"{self.collected_count}/{self.required_count}"
                )
            return

        if self.session.should_sample(sampled_at):
            self.session.add(aligned_face, sampled_at)
            print(
                "Verification collected: "
                f"{self.collected_count}/{self.required_count}"
            )

    def _render_frame(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
    ) -> None:
        """Build UI data and display the current camera frame."""
        render_state = CameraRenderStateBuilder.build(
            analysis=analysis,
            collected_count=self.collected_count,
            required_count=self.required_count,
            verification_result=self.verification_result,
        )
        rendered_frame = CameraRenderer.render(frame=frame, state=render_state)
        cv2.imshow(self.WINDOW_NAME, rendered_frame)

    def _handle_completed_collection(self) -> bool:
        """Run the correct action after the active session has enough samples."""
        if self.is_enrollment_mode:
            if self.enrollment_collector.is_complete:
                self._handle_completed_enrollment()
                return True
            return False

        if self.session.is_complete:
            self._handle_completed_verification()

        return False

    def _handle_completed_enrollment(self) -> None:
        """Train and save the owner model after enrollment is complete."""
        if self.enrollment_service is None:
            raise RuntimeError(
                "EnrollmentService is required in enrollment mode."
            )

        print("Enrollment completed")
        self.enrollment_service.enroll(
            aligned_faces=self.enrollment_collector.samples,
        )

    def _handle_completed_verification(self) -> None:
        """Verify the collected frames and prepare a new verification round."""
        if self.verification_service is None:
            raise RuntimeError(
                "FaceVerificationService is required in verification mode."
            )

        print("Verification samples collected")
        is_match, mean_error, frame_errors = self.verification_service.verify(
            aligned_faces=self.session.get_frames(),
        )
        self.verification_result = is_match

        print(f"Mean reconstruction error: {mean_error:.4f}")
        print(f"Frame errors: {frame_errors}")
        print(
            "Verification successful: owner matched."
            if is_match
            else "Verification failed: owner not matched."
        )
        self.session.reset()

    def _reset_timed_out_verification(self, now: float) -> None:
        """Discard an incomplete verification session after its timeout."""
        if not self.is_enrollment_mode and self.session.has_timed_out(now):
            print("Verification session timed out. Resetting...")
            self.session.reset()
            self.verification_result = None

    def _handle_keyboard(self) -> bool:
        """Handle exit and manual-reset keyboard commands."""
        key = cv2.waitKey(1) & 0xFF

        if self._should_stop(key):
            return True

        if key == ord("r"):
            print("Collection manually reset.")
            self._reset_active_collection()

        return False

    def _should_stop(self, key: int) -> bool:
        """Return whether the user pressed exit or closed the window."""
        if key in (ord("q"), 27):
            return True

        return cv2.getWindowProperty(
            self.WINDOW_NAME,
            cv2.WND_PROP_VISIBLE,
        ) < 1

    def _reset_active_collection(self) -> None:
        """Clear samples in the mode that is currently active."""
        if self.is_enrollment_mode:
            self.enrollment_collector.reset()
        else:
            self.session.reset()

        self.verification_result = None

    @property
    def is_enrollment_mode(self) -> bool:
        """Return whether this runtime is collecting enrollment samples."""
        return self.enrollment_collector is not None

    @property
    def collected_count(self) -> int:
        """Return the number of samples collected in the active mode."""
        if self.is_enrollment_mode:
            return len(self.enrollment_collector.samples)
        return self.session.collected_count

    @property
    def required_count(self) -> int:
        """Return the number of samples required in the active mode."""
        if self.is_enrollment_mode:
            return self.enrollment_collector.required_samples
        return self.session.required_frames