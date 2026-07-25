from src.camera.camera_input import CameraInput
from src.camera.camera_runtime import CameraRuntime
from src.detection.yunet_detector import (
    YuNetFaceDetector,
)
from src.capture.frame_session import FrameSession
from src.selection.rule_based_face_selector import RuleBasedFaceSelector
from src.validation.face_sample_validator import FaceSampleValidator

camera = CameraInput(camera_index=0)

# adding detector
face_detector = YuNetFaceDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
)

frames = FrameSession()

# adding selector
selector = RuleBasedFaceSelector(
    area_weight=0.6,
    center_weight=0.4,
    ambiguity_margin=0.15,
)

# adding validator
face_validator = FaceSampleValidator(
    min_face_area_ratio=0.08,
    min_edge_margin_ratio=0.05,
    min_blur_score=80.0,
)

runtime = CameraRuntime(
    camera=camera,
    face_detector=face_detector,
    session=frames,
    face_selector = selector,
    face_validator=face_validator
)

runtime.run()