"""Build the data needed by the camera renderer."""

from src.camera.models import FrameAnalysis
from src.draw.models import CameraRenderState
from src.selection.models import FaceSelectionResult, SelectionStatus


class CameraRenderStateBuilder:
    """Converts frame-analysis results into renderer input."""

    @staticmethod
    def build(
        analysis: FrameAnalysis,
        collected_count: int,
        required_count: int,
        verification_result: bool | None,
    ) -> CameraRenderState:
        """Create one render state for the current camera frame."""
        status, status_color = (
            CameraRenderStateBuilder.get_selection_status(
                selection=analysis.selection,
                is_valid_sample=analysis.is_sample_ready,
            )
        )

        alignment_result = analysis.alignment_result
        selected_face = analysis.selected_face

        return CameraRenderState(
            detections=analysis.detections,
            selected_face=selected_face,
            face_crop=(
                alignment_result.image
                if alignment_result is not None
                else None
            ),
            crop_landmarks=(
                alignment_result.landmarks
                if alignment_result is not None
                else None
            ),
            original_landmarks=(
                selected_face.landmarks
                if selected_face is not None
                else None
            ),
            is_valid_sample=analysis.is_sample_ready,
            collected_count=collected_count,
            required_frames=required_count,
            status=status,
            status_color=status_color,
            verification_result=verification_result,
        )

    @staticmethod
    def get_selection_status(
        selection: FaceSelectionResult,
        is_valid_sample: bool,
    ) -> tuple[str, tuple[int, int, int]]:
        """Return a short UI message and its BGR color."""
        if selection.status is SelectionStatus.NO_FACE:
            return "No face detected", (0, 0, 255)

        if selection.status is SelectionStatus.AMBIGUOUS:
            return "Cannot determine target face", (0, 165, 255)

        if not is_valid_sample:
            return "Face selected, but sample is invalid", (0, 165, 255)

        return "Target face is ready", (0, 255, 0)