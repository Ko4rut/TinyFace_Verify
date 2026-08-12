from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.camera.camera_input import CameraInput


CAMERA_PATCH = "src.camera.camera_input.cv2.VideoCapture"


def test_init_opens_requested_camera() -> None:
    with patch(CAMERA_PATCH) as video_capture:
        camera_input = CameraInput(camera_index=2)

    video_capture.assert_called_once_with(2)
    assert camera_input.camera_index == 2


def test_face_from_path_returns_message_without_path() -> None:
    with patch(CAMERA_PATCH):
        camera_input = CameraInput()

    assert camera_input.face_from_path() == "Image_path is None"


def test_face_from_path_reads_configured_image() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with (
        patch(CAMERA_PATCH),
        patch(
            "src.camera.camera_input.cv2.imread",
            return_value=image,
        ) as imread,
    ):
        camera_input = CameraInput(image_path="images/face.jpg")
        result = camera_input.face_from_path()

    imread.assert_called_once_with(str(Path("images/face.jpg")))
    assert result is image


def test_face_from_camera_yields_frame_and_releases_on_generator_close() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    device = Mock()
    device.isOpened.return_value = True
    device.read.return_value = True, frame

    with patch(CAMERA_PATCH, return_value=device):
        camera_input = CameraInput()
        frames = camera_input.face_from_camera()
        assert next(frames) is frame
        frames.close()

    device.release.assert_called_once_with()
    assert camera_input.camera is None


def test_face_from_camera_raises_when_device_cannot_open() -> None:
    device = Mock()
    device.isOpened.return_value = False

    with patch(CAMERA_PATCH, return_value=device):
        camera_input = CameraInput()
        frames = camera_input.face_from_camera()
        with pytest.raises(RuntimeError, match="Unable to open camera: 0"):
            next(frames)


def test_face_from_camera_raises_and_releases_when_read_fails() -> None:
    device = Mock()
    device.isOpened.return_value = True
    device.read.return_value = False, None

    with patch(CAMERA_PATCH, return_value=device):
        camera_input = CameraInput()
        frames = camera_input.face_from_camera()
        with pytest.raises(RuntimeError, match="Unable to capture camera"):
            next(frames)

    device.release.assert_called_once_with()
    assert camera_input.camera is None


def test_close_is_idempotent() -> None:
    device = Mock()

    with patch(CAMERA_PATCH, return_value=device):
        camera_input = CameraInput()

    camera_input.close()
    camera_input.close()

    device.release.assert_called_once_with()
