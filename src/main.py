from src.camera.camera_input import CameraInput
from src.camera.camera_runtime_service import CameraRuntimeService
from src.detection.yunet_detector import (
    YuNetFaceDetector,
)
from src.capture.frame_session import FrameSession
from src.selection.rule_based_face_selector import RuleBasedFaceSelector
from src.validation.face_sample_validator import FaceSampleValidator
from src.alignment.face_aligner import FaceAligner
from src.enrollment.enrollment_sample_collector import EnrollmentSampleCollector
from src.enrollment.enrollment_service import EnrollmentService
from src.eigenface.preprocessor import FacePreprocessor
from src.eigenface.eigenface_model import EigenfaceModel
from src.verification.face_verification_service import FaceVerificationService

system_mode = "Enrollment"
system_mode = "Verify"

camera = CameraInput(camera_index=0)

# adding detector
face_detector = YuNetFaceDetector(
    model_path=(
        "src/"
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
)

frames = FrameSession()
preprocessor = FacePreprocessor()

# adding selector
selector = RuleBasedFaceSelector(
    area_weight=0.5,
    center_weight=0.3,
    ambiguity_margin=0.1,
)

# adding validator
face_validator = FaceSampleValidator(
    min_face_area_ratio=0.02,
    min_edge_margin_ratio=0.05,
    min_blur_score=80.0,
)
face_aligner = FaceAligner(output_size=(112, 112))

enrollment_collector = EnrollmentSampleCollector(20,0.3)

eigenface = EigenfaceModel(n_components=9)

verify_service = FaceVerificationService(
    preprocessor=preprocessor,
    model_path=(
        "src/"
        "models/"
        "templates/"
        "owner_template.pkl"
    ),
    error_threshold= 0.03
)

enrollment_service = EnrollmentService(
    preprocessor=preprocessor,
    eigenface_model=eigenface,
    model_path=(
        "src/"
        "models/"
        "templates/"
        "owner_template.pkl"
    ))

enrollment_collector = None
if system_mode == "Enrollment":
    enrollment_collector = EnrollmentSampleCollector(20,0.3)


runtime = CameraRuntimeService(
    camera=camera,
    face_detector=face_detector,
    session=frames,
    face_selector = selector,
    face_validator=face_validator,
    face_aligner=face_aligner,
    enrollment_collector=enrollment_collector,
    enrollment_service= enrollment_service,
    verification_service=verify_service
)

runtime.run()