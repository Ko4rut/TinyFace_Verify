from src.detection.models import FaceDetection

def compute_width_height(face: FaceDetection) -> tuple[int,int]:
    x1, y1, x2, y2 = face.bbox
    width_face = abs(x1-x2)
    height_face =abs(y1-y2)
    return width_face, height_face

