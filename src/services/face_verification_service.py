from pathlib import Path
import pickle

import numpy as np

from src.eigenface.eigenface_model import EigenfaceModel
from src.eigenface.preprocessor import FacePreprocessor


class FaceVerificationService:
    """
    Verifies captured faces using the owner-specific
    PCA/Eigenface model created during enrollment.
    """

    def __init__(
        self,
        preprocessor: FacePreprocessor,
        model_path: str | Path,
        error_threshold: float,
    ) -> None:
        if error_threshold < 0:
            raise ValueError(
                "error_threshold must be non-negative."
            )

        self.preprocessor = preprocessor
        self.model_path = Path(model_path)
        self.error_threshold = error_threshold

    def verify(
        self,
        aligned_faces: list[np.ndarray],
    ) -> tuple[bool, float, np.ndarray]:
        """
        Returns:
            is_match:
                True when mean reconstruction error
                is below the configured threshold.

            mean_error:
                Mean reconstruction error of the session.

            frame_errors:
                Reconstruction error for every frame.
        """
        if not aligned_faces:
            raise ValueError(
                "At least one aligned face is required."
            )

        eigenface_model = self._load_model()

        face_vectors = np.stack(
            [
                self.preprocessor.preprocess(face)
                for face in aligned_faces
            ],
            axis=0,
        ).astype(np.float32)

        embeddings = eigenface_model.transform_batch(
            face_vectors,
        )

        reconstructed_vectors = (
            embeddings @ eigenface_model.components
            + eigenface_model.mean_face
        )

        frame_errors = np.mean(
            (face_vectors - reconstructed_vectors) ** 2,
            axis=1,
        )

        mean_error = float(np.mean(frame_errors))

        is_match = mean_error <= self.error_threshold

        return is_match, mean_error, frame_errors

    def _load_model(self) -> EigenfaceModel:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Owner PCA model not found: "
                f"{self.model_path.resolve()}"
            )

        with self.model_path.open("rb") as file:
            payload = pickle.load(file)

        model = EigenfaceModel(
            n_components=payload["n_components"],
        )

        model.mean_face = np.asarray(
            payload["mean_face"],
            dtype=np.float32,
        )

        model.components = np.asarray(
            payload["components"],
            dtype=np.float32,
        )

        return model