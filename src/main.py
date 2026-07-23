from src.camera.camera_input import CameraInput
from src.camera.camera_runtime import CameraRuntime
from src.detection.yunet_detector import (
    YuNetFaceDetector,
)
from src.capture.frame_session import FrameSession
from src.selection.rule_based_face_selector import RuleBasedFaceSelector

camera = CameraInput(camera_index=0)

face_detector = YuNetFaceDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
)

frames = FrameSession()

selector = RuleBasedFaceSelector(
    area_weight=0.6,
    center_weight=0.4,
    ambiguity_margin=0.15,
)

runtime = CameraRuntime(
    camera=camera,
    face_detector=face_detector,
    session=frames,
    face_selector = selector
)

runtime.run()