import numpy as np


class EigenfaceModel:
    """
    Learns an Eigenfaces/PCA projection from training face vectors.
    """

    def __init__(
        self,
        n_components: int = 100,
    ) -> None:
        if n_components <= 0:
            raise ValueError(
                "n_components must be positive."
            )

        self.n_components = n_components

        self.mean_face: np.ndarray | None = None
        self.components: np.ndarray | None = None

    def fit(
        self,
        training_vectors: np.ndarray,
    ) -> None:
        """
        Fits PCA using face vectors.

        Input:
            training_vectors:
                Shape (n_samples, n_features).
                Example: (400, 12544) for 112 x 112 grayscale faces.
        """
        if training_vectors.ndim != 2:
            raise ValueError(
                "training_vectors must be a 2D array."
            )

        n_samples, n_features = training_vectors.shape

        if n_samples < 2:
            raise ValueError(
                "At least two training samples are required."
            )

        max_components = min(n_samples, n_features)

        if self.n_components > max_components:
            raise ValueError(
                "n_components cannot be greater than "
                f"{max_components}."
            )

        training_vectors = training_vectors.astype(
            np.float32,
            copy=False,
        )

        # Mean face: shape (n_features,)
        self.mean_face = np.mean(
            training_vectors,
            axis=0,
        )

        centered_vectors = (
            training_vectors - self.mean_face
        )

        # Vt shape: (min(n_samples, n_features), n_features)
        _, _, vt = np.linalg.svd(
            centered_vectors,
            full_matrices=False,
        )

        # Each row is one principal component / eigenface.
        self.components = vt[:self.n_components]

    def transform(
        self,
        face_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Projects one face vector into Eigenface/PCA space.

        Input:
            face_vector shape (n_features,).

        Returns:
            Embedding shape (n_components,).
        """
        if face_vector.ndim != 1:
            raise ValueError(
                "face_vector must be a 1D array."
            )

        self._ensure_fitted()

        if face_vector.shape[0] != self.mean_face.shape[0]:
            raise ValueError(
                "face_vector has an unexpected feature size."
            )

        centered_vector = (
            face_vector.astype(np.float32, copy=False)
            - self.mean_face
        )

        return centered_vector @ self.components.T

    def transform_batch(
        self,
        face_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Projects multiple face vectors into Eigenface/PCA space.

        Input:
            face_vectors shape (n_samples, n_features).

        Returns:
            Embeddings shape (n_samples, n_components).
        """
        if face_vectors.ndim != 2:
            raise ValueError(
                "face_vectors must be a 2D array."
            )

        self._ensure_fitted()

        if face_vectors.shape[1] != self.mean_face.shape[0]:
            raise ValueError(
                "face_vectors have an unexpected feature size."
            )

        centered_vectors = (
            face_vectors.astype(np.float32, copy=False)
            - self.mean_face
        )

        return centered_vectors @ self.components.T

    def _ensure_fitted(self) -> None:
        """Raises an error when the model has not been fitted."""
        if self.mean_face is None or self.components is None:
            raise RuntimeError(
                "EigenfaceModel must be fitted before transform."
            )