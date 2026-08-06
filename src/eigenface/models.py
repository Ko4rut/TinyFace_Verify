from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class TrainingData:
    features: np.ndarray  # (n_samples, 12544)
    labels: np.ndarray    # (n_samples,)
    image_paths: list[Path]