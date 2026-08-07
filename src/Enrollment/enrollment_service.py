class EnrollmentSampleCollector:
    def __init__(
        self,
        required_samples: int = 20,
        sample_interval_seconds: float = 0.3,
    ) -> None:
        ...

    def try_add(
        self,
        aligned_face: np.ndarray,
        sampled_at: float,
    ) -> bool:
        """
        Thêm một ảnh aligned nếu:
        - chưa đủ số lượng
        - đã qua sample interval

        Return:
            True  -> đã nhận ảnh
            False -> bỏ qua ảnh
        """

    def is_complete(self) -> bool:
        ...

    def get_samples(self) -> list[np.ndarray]:
        ...