import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.camera_input import CameraInput
from pathlib import Path

def test_init_opens_correct_camera_index():
    with patch("src.camera_input.cv2.VideoCapture") as mock_video_capture:
        camera_input = CameraInput(camera_index=2)

    mock_video_capture.assert_called_once_with(2)
    assert camera_input.camera_index == 2   
    
def test_face_from_path_returns_message_when_path_is_none():
    with patch("src.camera_input.cv2.VideoCapture"):
        camera_input = CameraInput(image_path=None)

    result = camera_input.face_from_path()

    assert result == "Image_path is None"

def test_face_from_path_reads_correct_image():
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with (
        patch("src.camera_input.cv2.VideoCapture"),
        patch(
            "src.camera_input.cv2.imread",
            return_value=fake_image,
        ) as mock_imread,
    ):
        camera_input = CameraInput(image_path="images/face.jpg")
        result = camera_input.face_from_path()

    mock_imread.assert_called_once_with(
        str(Path("images/face.jpg"))
    )
    assert result is fake_image

def test_face_from_camera_yields_frame():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_camera = Mock()
    mock_camera.isOpened.return_value = True
    mock_camera.read.return_value = (True, fake_frame)

    with patch(
        "src.camera_input.cv2.VideoCapture",
        return_value=mock_camera,
    ):
        camera_input = CameraInput()
        frame_generator = camera_input.face_from_camera()

        result = next(frame_generator)

        frame_generator.close()

    assert result is fake_frame
    
def test_face_from_camera_raises_when_camera_cannot_open():
    mock_camera = Mock()
    mock_camera.isOpened.return_value = False

    with patch(
        "src.camera_input.cv2.VideoCapture",
        return_value=mock_camera,
    ):
        camera_input = CameraInput()
        frame_generator = camera_input.face_from_camera()

        with pytest.raises(
            RuntimeError,
            match="Unable to open camera: 0",
        ):
            next(frame_generator)
            
def test_face_from_camera_raises_when_read_fails():
    mock_camera = Mock()
    mock_camera.isOpened.return_value = True
    mock_camera.read.return_value = (False, None)

    with patch(
        "src.camera_input.cv2.VideoCapture",
        return_value=mock_camera,
    ):
        camera_input = CameraInput()
        frame_generator = camera_input.face_from_camera()

        with pytest.raises(
            RuntimeError,
            match="Unable to capture camera",
        ):
            next(frame_generator)

    mock_camera.release.assert_called_once()
    assert camera_input.camera is None
    
def test_face_from_camera_yields_frame():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_camera = Mock()
    mock_camera.isOpened.return_value = True
    mock_camera.read.return_value = (True, fake_frame)

    with patch(
        "src.camera_input.cv2.VideoCapture",
        return_value=mock_camera,
    ):
        camera_input = CameraInput()
        frame_generator = camera_input.face_from_camera()

        result = next(frame_generator)

        frame_generator.close()

    assert result is fake_frame
    mock_camera.release.assert_called_once()
    assert camera_input.camera is None
    
def test_close_releases_camera():
    mock_camera = Mock()

    with patch(
        "src.camera_input.cv2.VideoCapture",
        return_value=mock_camera,
    ):
        camera_input = CameraInput()

    camera_input.close()

    mock_camera.release.assert_called_once()
    assert camera_input.camera is None  
    
def test_close_can_be_called_twice():
    mock_camera = Mock()

    with patch(
        "src.camera_input.cv2.VideoCapture",
        return_value=mock_camera,
    ):
        camera_input = CameraInput()

    camera_input.close()
    camera_input.close()

    mock_camera.release.assert_called_once()
    assert camera_input.camera is None