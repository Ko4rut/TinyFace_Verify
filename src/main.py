from src.camera.camera_input import CameraInput
from src.camera.camera_runtime import CameraRuntime
from src.detection.yunet_detector import (
    YuNetFaceDetector,
)
from src.capture.frame_session import FrameSession


camera = CameraInput(camera_index=0)

face_detector = YuNetFaceDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
)

frames = FrameSession()

runtime = CameraRuntime(
    camera=camera,
    face_detector=face_detector,
    session=frames,
)

runtime.run()