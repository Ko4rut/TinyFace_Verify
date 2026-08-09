from pathlib import Path
import pickle

import numpy as np

from src.eigenface.preprocessor import FacePreprocessor


class FaceVerificationService:
    """
    Baseline 1:1 face verification service.

    Creates one face template from enrollment images
    and saves it for later verification.
    """

    def __init__(
        self,
        preprocessor: FacePreprocessor,
        template_path: str | Path,
    ) -> None:
        self.preprocessor = preprocessor
        self.template_path = Path(template_path)

    def enroll(
        self,
        aligned_faces: list[np.ndarray],
    ) -> np.ndarray:
        """
        Creates and saves one template from enrollment faces.
        """
        if not aligned_faces:
            raise ValueError(
                "At least one aligned face is required."
            )

        face_vectors = np.stack(
            [
                self.preprocessor.preprocess(face)
                for face in aligned_faces
            ],
            axis=0,
        )

        normalized_vectors = np.stack(
            [
                self._normalize_vector(vector)
                for vector in face_vectors
            ],
            axis=0,
        )

        template = np.mean(
            normalized_vectors,
            axis=0,
        )

        template = self._l2_normalize(template)

        self._save_template(
            template=template,
            sample_count=len(aligned_faces),
        )

        return template

    def _save_template(
        self,
        template: np.ndarray,
        sample_count: int,
    ) -> None:
        self.template_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "template": template,
            "sample_count": sample_count,
        }

        with self.template_path.open("wb") as file:
            pickle.dump(payload, file)

    @staticmethod
    def _normalize_vector(
        vector: np.ndarray,
    ) -> np.ndarray:
        """
        Reduces overall brightness influence per image.
        """
        centered_vector = vector - np.mean(vector)

        return FaceVerificationService._l2_normalize(
            centered_vector,
        )

    @staticmethod
    def _l2_normalize(
        vector: np.ndarray,
    ) -> np.ndarray:
        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError(
                "Cannot normalize a zero vector."
            )

        return vector / norm