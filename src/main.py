from src.camera_input import CameraInput
from src.camera_runtime import CameraRuntime


if __name__ == "__main__":
    camera = CameraInput(camera_index=0)

    runtime = CameraRuntime(
        camera=camera,
        required_frames=5,
        sample_interval=0.15,
        session_timeout=2.0,
    )

    runtime.run()