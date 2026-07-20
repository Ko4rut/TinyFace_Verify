from src.camera_input import CameraInput
from src.camera_runtime import CameraRuntime
from src.preprocessing.detectors.yunet_detector import (
    YuNetFaceDetector,
)


camera = CameraInput(camera_index=0)

face_detector = YuNetFaceDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
)

runtime = CameraRuntime(
    camera=camera,
    face_detector=face_detector,
    required_frames=5,
)

runtime.run()