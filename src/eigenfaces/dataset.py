from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np

class AttFaceDataset:
    SUPPORTED_EXTENSION = {".pgm"}
    
    def __init__(self, root_directory: str | Path) -> None:
        self.root_directory = Path(root_directory)
    
        if not self.root_directory.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: "
                f"{self.root_directory}"
            )
            
        if not self.root_directory.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: "
                f"{self.root_directory}"
            )
            
    def iter_samples(
        self,
    ) -> Iterator[tuple[np.ndarray, int, Path]]:
        """
        Yields:
            image:
                Ảnh raw đọc từ dataset.

            label:
                ID danh tính lấy từ tên thư mục.
                Ví dụ s1 -> 1.

            image_path:
                Đường dẫn ảnh, phục vụ debug hoặc lưu processed.
        """
        for person_directory in self._get_person_directories():
            label = self._parse_label(person_directory.name)

            for image_path in self._get_image_paths(person_directory):
                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_UNCHANGED,
                )

                if image is None or image.size == 0:
                    raise ValueError(
                        f"Cannot read image: {image_path}"
                    )

                yield image, label, image_path