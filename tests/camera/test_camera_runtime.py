from unittest.mock import Mock, patch

import numpy as np

from src.services.camera_runtime_service import CameraRuntimeService
from src.capture.frame_session import FrameSession
from src.detection.models import FaceDetection
from src.selection.models import FaceSelectionResult, SelectionStatus


def make_runtime(
    *,
    session: FrameSession | None = None,
    enrollment_collector=None,
) -> tuple[CameraRuntimeService, Mock, Mock, Mock, Mock, Mock]:
    """Create a runtime with all external collaborators mocked."""
    camera = Mock()
    detector = Mock()
    selector = Mock()
    validator = Mock()
    aligner = Mock()
    enrollment_service = Mock()
    verification_service = Mock()

    runtime = CameraRuntimeService(
        camera=camera,
        face_detector=detector,
        session=session or FrameSession(),
        face_selector=selector,
        face_validator=validator,
        face_aligner=aligner,
        enrollment_collector=enrollment_collector,
        enrollment_service=enrollment_service,
        verification_service=verification_service,
    )

    return (
        runtime,
        camera,
        detector,
        selector,
        validator,
        aligner,
    )


def make_face() -> FaceDetection:
    return FaceDetection(
        bbox=(10, 10, 40, 40),
        confidence=0.99,
        landmarks=np.array(
            [[15, 18], [35, 18], [25, 25], [18, 33], [32, 33]],
            dtype=np.float32,
        ),
    )


@patch("src.services.camera_runtime_service.cv2.destroyAllWindows")
@patch("src.services.camera_runtime_service.cv2.waitKey", return_value=ord("q"))
@patch("src.services.camera_runtime_service.cv2.imshow")
@patch("src.services.camera_runtime_service.cv2.flip", side_effect=lambda frame, _: frame)
@patch("src.services.camera_runtime_service.CameraRenderer.render")
def test_run_renders_no_face_frame_and_closes_camera(
    mock_render: Mock,
    _mock_flip: Mock,
    mock_imshow: Mock,
    _mock_wait_key: Mock,
    mock_destroy_windows: Mock,
) -> None:
    runtime, camera, detector, selector, validator, aligner = make_runtime()
    raw_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    rendered_frame = np.ones((20, 30, 3), dtype=np.uint8)
    camera.face_from_camera.return_value = iter([raw_frame])
    detector.detect.return_value = []
    selector.select.return_value = FaceSelectionResult(SelectionStatus.NO_FACE)
    mock_render.return_value = rendered_frame

    runtime.run()

    detector.detect.assert_called_once_with(raw_frame)
    selector.select.assert_called_once_with(detections=[], frame_shape=raw_frame.shape)
    validator.validate.assert_not_called()
    aligner.align.assert_not_called()
    mock_imshow.assert_called_once_with(runtime.WINDOW_NAME, rendered_frame)
    camera.close.assert_called_once_with()
    mock_destroy_windows.assert_called_once_with()


def test_analyze_frame_does_not_align_invalid_selected_face() -> None:
    runtime, _, detector, selector, validator, aligner = make_runtime()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    face = make_face()
    detector.detect.return_value = [face]
    selector.select.return_value = FaceSelectionResult(SelectionStatus.SELECTED, face)
    validator.validate.return_value = False

    analysis = runtime._analyze_frame(frame)

    assert analysis.selected_face is face
    assert analysis.alignment_result is None
    assert analysis.is_sample_ready is False
    validator.validate.assert_called_once_with(frame=frame, face=face)
    aligner.align.assert_not_called()


def test_completed_verification_resets_session_and_records_result() -> None:
    session = FrameSession(required_frames=1)
    runtime, _, _, _, _, _ = make_runtime(session=session)
    sample = np.zeros((112, 112, 3), dtype=np.uint8)
    session.add(sample, now=5.0)
    runtime.verification_service.verify.return_value = (
        True,
        0.01,
        np.array([0.01], dtype=np.float32),
    )

    runtime._handle_completed_verification()

    runtime.verification_service.verify.assert_called_once()
    assert runtime.verification_result is True
    assert session.collected_count == 0


def test_completed_enrollment_requires_enrollment_service() -> None:
    runtime, _, _, _, _, _ = make_runtime(enrollment_collector=Mock())
    runtime.enrollment_service = None

    try:
        runtime._handle_completed_enrollment()
    except RuntimeError as error:
        assert str(error) == "EnrollmentService is required in enrollment mode."
    else:
        raise AssertionError("Expected enrollment without service to fail")
