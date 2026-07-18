from camera_input import CameraInput
import cv2

if __name__ == "__main__":
    camera = CameraInput("../data/avt.jpg",0)
    face = camera.face_from_path()
    cv2.imshow("face",face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    