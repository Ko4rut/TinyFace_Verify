import cv2
import numpy as np

from src.detection.models import FaceDetection
from src.draw.face_renderer import FaceRenderer
from src.draw.face_panel_renderer import FacePanelRenderer
from src.draw.models import CameraRenderState

class CameraRenderer:
    @staticmethod
    def render(
        frame: np.ndarray,
        state: CameraRenderState,
    ) -> np.ndarray:
        preview = frame.copy()

        preview = FaceRenderer.draw(
            frame=preview,
            detections=state.detections,
            selected_face=state.selected_face,
            required_frames= state.required_frames,
            collected_count= state.collected_count,
            status=state.status,
            status_color=state.status_color,
            verification_result=state.verification_result,
        )
        
    

        panel = FacePanelRenderer.draw(
            frame_width=preview.shape[1],
            face_crop=state.face_crop,
            face_landmarks=state.crop_landmarks,
            is_valid_sample=state.is_valid_sample,
            # collected_count=state.collected_count,
            # required_frames=state.required_frames,
            original_landmarks=state.original_landmarks,
            status=state.status,
            # status_color=state.status_color,
        )

        return np.vstack((preview, panel))
    
    