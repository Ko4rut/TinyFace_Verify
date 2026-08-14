import numpy as np
import pytest

from src.evaluation.eigenface_evaluator import EigenfaceVerificationEvaluator


def test_evaluation_separates_genuine_and_impostor_vectors() -> None:
    evaluator = EigenfaceVerificationEvaluator(n_components=1)

    evaluation = evaluator.evaluate(
        enrollment_vectors=np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        ),
        genuine_vectors=np.array(
            [[0.5, 0.0], [1.5, 0.0]],
            dtype=np.float32,
        ),
        impostor_vectors=np.array(
            [[0.5, 2.0], [1.5, -2.0]],
            dtype=np.float32,
        ),
    )

    threshold = evaluation.recommended_threshold

    assert evaluation.genuine_sample_count == 2
    assert evaluation.impostor_sample_count == 2
    assert np.all(evaluation.genuine_errors < 1e-6)
    assert np.all(evaluation.impostor_errors > 1.0)
    assert threshold.false_accept_rate == 0.0
    assert threshold.false_reject_rate == 0.0


def test_evaluation_rejects_mismatched_feature_sizes() -> None:
    evaluator = EigenfaceVerificationEvaluator(n_components=1)

    with pytest.raises(
        ValueError,
        match="genuine_vectors has an unexpected feature size",
    ):
        evaluator.evaluate(
            enrollment_vectors=np.zeros((2, 2), dtype=np.float32),
            genuine_vectors=np.zeros((1, 3), dtype=np.float32),
            impostor_vectors=np.zeros((1, 2), dtype=np.float32),
        )
