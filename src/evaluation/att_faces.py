"""ATT Faces benchmark adapter for offline Eigenface evaluation."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eigenface.dataset import AttFaceDataset
from src.eigenface.preprocessor import FacePreprocessor
from src.evaluation.eigenface_evaluator import EigenfaceVerificationEvaluator
from src.evaluation.models import VerificationEvaluation


class AttFacesBenchmark:
    """Evaluate one owner against the remaining ATT Faces identities."""

    def __init__(
        self,
        dataset_root: str | Path,
        preprocessor: FacePreprocessor,
        n_components: int,
        enrollment_samples: int = 5,
    ) -> None:
        if enrollment_samples < 2:
            raise ValueError("enrollment_samples must be at least two")

        self.dataset = AttFaceDataset(dataset_root)
        self.preprocessor = preprocessor
        self.enrollment_samples = enrollment_samples
        self.evaluator = EigenfaceVerificationEvaluator(n_components)

    def evaluate_owner(self, owner_label: int) -> VerificationEvaluation:
        """Use the owner's first samples for enrollment and later samples to test."""
        grouped_vectors = self._load_vectors_by_label()

        if owner_label not in grouped_vectors:
            raise ValueError(f"Owner label {owner_label} does not exist")

        owner_vectors = grouped_vectors.pop(owner_label)
        if len(owner_vectors) <= self.enrollment_samples:
            raise ValueError(
                "Owner must have more samples than enrollment_samples"
            )

        enrollment_vectors = np.stack(
            owner_vectors[:self.enrollment_samples],
            axis=0,
        )
        genuine_vectors = np.stack(
            owner_vectors[self.enrollment_samples:],
            axis=0,
        )
        impostor_vectors = np.stack(
            [
                vector
                for vectors in grouped_vectors.values()
                for vector in vectors
            ],
            axis=0,
        )

        return self.evaluator.evaluate(
            enrollment_vectors=enrollment_vectors,
            genuine_vectors=genuine_vectors,
            impostor_vectors=impostor_vectors,
        )

    def _load_vectors_by_label(self) -> dict[int, list[np.ndarray]]:
        grouped_vectors: dict[int, list[np.ndarray]] = defaultdict(list)

        for image, label, _ in self.dataset.iter_samples():
            grouped_vectors[label].append(
                self.preprocessor.preprocess(image)
            )

        return dict(grouped_vectors)
