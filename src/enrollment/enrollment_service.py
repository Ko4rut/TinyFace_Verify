from pathlib import Path
import pickle

import numpy as np

from src.eigenface.eigenface_model import EigenfaceModel
from src.eigenface.preprocessor import FacePreprocessor


class EnrollmentService:
    """
    Fits an owner-specific Eigenface/PCA model
    from enrollment face samples.
    """

    def __init__(
        self,
        preprocessor: FacePreprocessor,
        eigenface_model: EigenfaceModel,
        model_path: str | Path,
    ) -> None:
        self.preprocessor = preprocessor
        self.eigenface_model = eigenface_model
        self.model_path = Path(model_path)

    def enroll(
        self,
        aligned_faces: list[np.ndarray],
    ) -> np.ndarray:
        """
        Fits PCA using enrolled owner face samples
        and saves the fitted model.

        Returns:
            Reconstruction errors of enrollment samples.
        """
        if len(aligned_faces) < 2:
            raise ValueError(
                "At least two aligned faces are required."
            )

        processed_faces = []
        
        for face in aligned_faces:
            processed_face = self.preprocessor.preprocess(face)
            processed_faces.append(processed_face)
        
        face_vectors = np.array(processed_faces)
        # face_vectors = np.stack(
        #     [
        #         self.preprocessor.preprocess(face)
        #         for face in aligned_faces
        #     ],
        #     axis=0,
        # ).astype(np.float32)

        self.eigenface_model.fit(face_vectors)

        reconstruction_errors = self._calculate_errors(
            face_vectors,
        )

        self._save_model(
            sample_count=len(aligned_faces),
            reconstruction_errors=reconstruction_errors,
        )

        return reconstruction_errors

    def _calculate_errors(
        self,
        face_vectors: np.ndarray,
    ) -> np.ndarray:
        embeddings = self.eigenface_model.transform_batch(
            face_vectors,
        )

        reconstructed_vectors = np.stack(
            [
                self.eigenface_model.inverse_transform(
                    embedding,
                )
                for embedding in embeddings
            ],
            axis=0,
        )

        return np.mean(
            (face_vectors - reconstructed_vectors) ** 2,
            axis=1,
        )

    def _save_model(
        self,
        sample_count: int,
        reconstruction_errors: np.ndarray,
    ) -> None:
        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "mean_face": self.eigenface_model.mean_face,
            "components": self.eigenface_model.components,
            "n_components": self.eigenface_model.n_components,
            "sample_count": sample_count,
            "enrollment_errors": reconstruction_errors,
        }

        with self.model_path.open("wb") as file:
            pickle.dump(payload, file)