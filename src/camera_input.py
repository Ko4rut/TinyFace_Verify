from pathlib import Path

import cv2
import numpy as np

class CameraInput:
    """Load image from camera to system"""
    
    def __init__(self, image_path: str = None, camera_index: int = 0):
        self.image_path = Path(image_path)
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)
        
    def face_from_path(self):
        """Load image from folder"""
        if self.image_path == None:
            return "Image_path is None"
        return cv2.imread(str(self.image_path))
    
    def face_from_camera(self):
        """Load from camera"""
        if not self.camera.isOpened():
            raise RuntimeError(f"Unable to open camera: {self.camera_index}")
        try: 
            success, frame = self.camera.read()
            
            if not success or frame is None:
                raise RuntimeError(f"Unable to capture camera")
            return frame
        except:
            print("Error, cannot use camer")
        finally:
            self.camera.release()