import numpy as np
class EigenfaceModel:
    def __init__(
        self,
        n_components: int = 100,
    ) -> None:
        ...
    
    def fit(
        self,
        training_vectors: np.ndarray,
    ) -> None:
        """
        Input: X shape (n_samples, 12544)
        """
        

    def transform(
        self,
        face_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Input:  vector shape (12544,)
        Output: embedding shape (n_components,)
        """

    def transform_batch(
        self,
        face_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Input:  X shape (n_samples, 12544)
        Output: embeddings shape (n_samples, n_components)
        """