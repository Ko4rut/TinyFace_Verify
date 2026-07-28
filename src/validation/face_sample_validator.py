from src.detection.models import FaceDetection
from src.utils.face_helper import compute_width_height,crop_face

import numpy as np 
import cv2
class FaceSampleValidator:
    def __init__(self, min_face_area_ratio: float, 
                min_edge_margin_ratio: float, min_blur_score: float) -> None:
        """
        face: The main input, which is validated before entering the preprocessing stage.
        min_face_area_ratio: The minimum ratio of the face area relative to the frame.
        min_edge_margin_ratio: The minimum margin between the frame edges and the face.
        min_blur_score: The minimum threshold used to evaluate the face's blurriness.
        """
        self.min_face_area_ratio = min_face_area_ratio
        self.min_edge_margin_ratio = min_edge_margin_ratio
        self.min_blur_score = min_blur_score
        
    def validate_face_area_ratio(self, face: FaceDetection, 
                                frame_width: int, frame_height: int) -> bool:
        """
        Validate ratio area criterion from fomular above
        """
        face_width, face_height = compute_width_height(face)
        result = (face_height*face_width)/(frame_height*frame_width)
        if result >= self.min_face_area_ratio:
            return True
        return False
    
    def face_edge_margin_ratio_validate(self, face: FaceDetection,
                               frame_width: int, frame_height: int) -> bool:
        """
        Validate edge criterion
        """
        x1, y1, x2, y2 = face.bbox
        margin_x = frame_width * self.min_edge_margin_ratio
        margin_y = frame_height * self.min_edge_margin_ratio
        if x1 < margin_x or y1 < margin_y or x2 > frame_width - margin_x or y2 > frame_height - margin_y:
            return False
        return True
    
    def validate_blur(
        self,
        frame: np.ndarray,
        face: FaceDetection,
    ) -> bool:
        face_crop = crop_face(frame, face).image

        if face_crop is None:
            return False

        gray_face = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY,
        )

        sharpness_score = cv2.Laplacian(
            gray_face,
            cv2.CV_64F,
        ).var()

        return sharpness_score >= self.min_blur_score
    
    def validate(self, frame: np.ndarray, face: FaceDetection) -> bool:
        if frame is None or frame.size == 0:
            return False

        frame_height, frame_width = frame.shape[:2]

        return (
            self.validate_face_area_ratio(
                face,
                frame_width,
                frame_height,
            )
            and self.face_edge_margin_ratio_validate(
                face,
                frame_width,
                frame_height,
            )
            and self.validate_blur(frame, face)
        )