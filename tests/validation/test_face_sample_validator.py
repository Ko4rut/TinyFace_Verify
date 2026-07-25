import numpy as np
import pytest

from src.detection.models import FaceDetection
from src.validation.face_sample_validator import FaceSampleValidator

def make_face(
    bbox: tuple[int, int, int, int],
) -> FaceDetection:
    return FaceDetection(
        bbox=bbox,
        confidence=0.95,
        landmarks=np.zeros((5, 2)),
    )
    
def test_face_area_ratio_returns_true_when_large_enough():
    validator = FaceSampleValidator(
        min_face_area_ratio=0.10,
        min_edge_margin_ratio=0.05,
        min_blur_score=80.0,
    )
    face = make_face((10, 10, 50, 50))

    result = validator.validate_face_area_ratio(
        face=face,
        frame_width=100,
        frame_height=100,
    )

    assert result is True
    
def test_edge_margin_returns_true_when_face_is_inside_margin():
    validator = FaceSampleValidator(
        min_face_area_ratio=0.10,
        min_edge_margin_ratio=0.10,
        min_blur_score=80.0,
    )
    face = make_face((10, 10, 90, 90))

    result = validator.face_edge_margin_ratio_validate(
        face=face,
        frame_width=100,
        frame_height=100,
    )

    assert result is True
    
@pytest.mark.parametrize(
    "bbox",
    [
        (9, 20, 80, 80),    # quá sát mép trái
        (20, 9, 80, 80),    # quá sát mép trên
        (20, 20, 91, 80),   # quá sát mép phải
        (20, 20, 80, 91),   # quá sát mép dưới
    ],
)
def test_edge_margin_returns_false_when_face_is_too_close_to_edge(
    bbox,
):
    validator = FaceSampleValidator(
        min_face_area_ratio=0.10,
        min_edge_margin_ratio=0.10,
        min_blur_score=80.0,
    )
    face = make_face(bbox)

    result = validator.face_edge_margin_ratio_validate(
        face=face,
        frame_width=100,
        frame_height=100,
    )

    assert result is False