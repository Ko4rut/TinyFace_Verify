import cv2
from pathlib import Path
from src.eigenface.dataset import AttFaceDataset
from src.eigenface.training_data_loader import DataLoader

a = DataLoader()
print(a.features)