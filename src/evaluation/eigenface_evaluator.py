"""Evaluate owner-specific Eigenface verification with held-out samples."""

import numpy as np

from src.eigenface.eigenface_model import EigenfaceModel
from src.evaluation.models import ThresholdMetrics, VerificationEvaluation


class EigenfaceVerificationEvaluator:
    """Fit one owner model and measure genuine versus impostor errors."""

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be greater than zero")

        self.n_components = n_components

    def evaluate(
        self,
        enrollment_vectors: np.ndarray,
        genuine_vectors: np.ndarray,
        impostor_vectors: np.ndarray,
    ) -> VerificationEvaluation:
        """Evaluate a PCA owner model using held-out genuine and impostor data.

        All vectors must already use the exact preprocessing used at runtime.
        A lower reconstruction error is considered a match.
        """
        enrollment = self._validate_vectors(
            vectors=enrollment_vectors,
            name="enrollment_vectors",
            min_samples=2,
        )
        genuine = self._validate_vectors(
            vectors=genuine_vectors,
            name="genuine_vectors",
            expected_feature_count=enrollment.shape[1],
        )
        impostor = self._validate_vectors(
            vectors=impostor_vectors,
            name="impostor_vectors",
            expected_feature_count=enrollment.shape[1],
        )

        model = EigenfaceModel(n_components=self.n_components)
        model.fit(enrollment)

        genuine_errors = self._reconstruction_errors(model, genuine)
        impostor_errors = self._reconstruction_errors(model, impostor)
        threshold_metrics = self._build_threshold_metrics(
            genuine_errors=genuine_errors,
            impostor_errors=impostor_errors,
        )

        return VerificationEvaluation(
            genuine_errors=genuine_errors,
            impostor_errors=impostor_errors,
            threshold_metrics=threshold_metrics,
            recommended_threshold=self._recommend_threshold(
                threshold_metrics,
            ),
        )

    @staticmethod
    def _validate_vectors(
        vectors: np.ndarray,
        name: str,
        min_samples: int = 1,
        expected_feature_count: int | None = None,
    ) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")

        if array.shape[0] < min_samples:
            raise ValueError(
                f"{name} must contain at least {min_samples} samples"
            )

        if array.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature")

        if expected_feature_count is not None and (
            array.shape[1] != expected_feature_count
        ):
            raise ValueError(f"{name} has an unexpected feature size")

        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

        return array

    @staticmethod
    def _reconstruction_errors(
        model: EigenfaceModel,
        vectors: np.ndarray,
    ) -> np.ndarray:
        embeddings = model.transform_batch(vectors)
        reconstructed = embeddings @ model.components + model.mean_face

        return np.mean((vectors - reconstructed) ** 2, axis=1)

    @staticmethod
    def _build_threshold_metrics(
        genuine_errors: np.ndarray,
        impostor_errors: np.ndarray,
    ) -> tuple[ThresholdMetrics, ...]:
        thresholds = np.unique(
            np.concatenate(
                (
                    np.array([0.0], dtype=np.float32),
                    genuine_errors,
                    impostor_errors,
                )
            )
        )

        return tuple(
            ThresholdMetrics(
                threshold=float(threshold),
                false_accept_rate=float(
                    np.mean(impostor_errors <= threshold)
                ),
                false_reject_rate=float(
                    np.mean(genuine_errors > threshold)
                ),
            )
            for threshold in thresholds
        )

    @staticmethod
    def _recommend_threshold(
        metrics: tuple[ThresholdMetrics, ...],
    ) -> ThresholdMetrics:
        return min(
            metrics,
            key=lambda metric: (
                abs(
                    metric.false_accept_rate
                    - metric.false_reject_rate
                ),
                metric.balanced_error_rate,
                metric.threshold,
            ),
        )
