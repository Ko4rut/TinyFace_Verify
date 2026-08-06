import cv2
import numpy as np

class DatasetPreprocessor:
    """
    Preprocessing data throught there step:
    - Convert to gray scale.
    - Resize image to 112x112.
    - Ensure type image is  float32
    - Normalize pixel to range [0.0, 1.0]
    - Flatten from ndarray [112, 112] to vector [12544,]
    """
    def __init__(self, output_size: tuple[int, int] = (112,112)) -> None:
        """
            output_size: Size of image processed
        """
        self.output_size = output_size
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        PreProcess workflow function.
        img: Image input to preproceesing
        """
        img_cvt_grayscale = self._to_grayscale(img)
        img_resized = self._resize(img_cvt_grayscale)
        img_cvt_float32 = self._to_float32(img_resized)
        img_Norm = self._normalize(img_cvt_float32)
        img_processed = self._flatten(img_Norm)
        return img_processed
    
    
    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Transfer to gray scale image.
        img: Image input to transfer.
        """
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def _resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to 112x112
        img: Image input to resize
        """
        return cv2.resize(img,self.output_size)
        
    def _to_float32(self, img: np.ndarray) -> np.ndarray:
        """
        Transfer to float 32 type to compute
        img: Image input to compute
        """
        return img.astype(np.float32)
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to range [0.0, 1.0]
        img: Image input to normalize
        """
        return img/255.0    
    
    def _flatten(self, img: np.ndarray) -> np.ndarray:
        """
        Flatten image to vector (12544,)
        img: Image input to flatten
        """
        return img.flatten()
        