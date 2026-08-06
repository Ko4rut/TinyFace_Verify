from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np

class AttFaceDataset:
    SUPPORTED_EXTENSIONS = {".pgm"}
    
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
            image:Image read from dataset.
            label: ID from dir.
            image_path: path.
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
                
    def _get_person_directories(self) -> list[Path]:
        directories = [
            path
            for path in self.root_directory.iterdir()
            if path.is_dir()
            and path.name.startswith("s")
            and path.name[1:].isdigit()
        ]

        return sorted(
            directories,
            key=lambda path: int(path.name[1:]),
        )

    def _get_image_paths(
        self,
        person_directory: Path,
    ) -> list[Path]:
        image_paths = [
            path
            for path in person_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
            )
        ]

        return sorted(
            image_paths,
            key=self._image_sort_key,
        )

    @staticmethod
    def _parse_label(directory_name: str) -> int:
        return int(directory_name[1:])

    @staticmethod
    def _image_sort_key(path: Path) -> tuple[int, str]:
        if path.stem.isdigit():
            return int(path.stem), path.name

        return 0, path.name

    def __len__(self) -> int:
        return sum(
            1
            for _ in self.iter_samples()
        )
        
