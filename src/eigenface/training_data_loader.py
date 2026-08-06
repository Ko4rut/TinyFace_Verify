import cv2
from pathlib import Path
import numpy as np
from src.eigenface.dataset import AttFaceDataset
from src.eigenface.preprocessor import DatasetPreprocessor
from src.eigenface.models import TrainingData

def DataLoader()->TrainingData:
    root_dir = Path("./data/raw/att_faces") 
    
    vectors = []
    labels = []
    paths  = []

    dataset = AttFaceDataset(root_dir)
    preprocessor = DatasetPreprocessor()
    
    for image, label, path in dataset.iter_samples():
        img_processed = preprocessor.preprocess(image)
        vectors.append(img_processed)
        labels.append(label)
        paths.append(path)
    
    X = np.stack(vectors, axis=0)
    data = TrainingData(features=X, labels= labels,image_paths=paths)
    return data