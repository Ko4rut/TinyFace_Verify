from camera_input import CameraInput
import cv2

if __name__ == "__main__":
    camera = CameraInput("../data/avt.jpg",0)
    # face = camera.face_from_path()
    # cv2.imshow("face",face)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for frame in camera.face_from_camera():
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("\x1b"):
            break

    cv2.destroyAllWindows()
    