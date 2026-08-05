from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eigenface.dataset import AttFaceDataset

def test_iter_samples_returns_image_label_and_path(
    tmp_path: Path,
) -> None:
    person_directory = tmp_path / "s1"
    person_directory.mkdir()

    expected_image = np.full(
        shape=(112, 92),
        fill_value=128,
        dtype=np.uint8,
    )

    image_path = person_directory / "1.pgm"

    success = cv2.imwrite(
        str(image_path),
        expected_image,
    )

    assert success

    dataset = AttFaceDataset(tmp_path)

    # Act
    samples = list(dataset.iter_samples())

    # Assert
    assert len(samples) == 1

    actual_image, actual_label, actual_path = samples[0]

    assert actual_label == 1
    assert actual_path == image_path
    assert actual_image.shape == (112, 92)
    assert actual_image.dtype == np.uint8
    assert np.array_equal(actual_image, expected_image)
    
def test_iter_samples_returns_samples_in_numeric_order(
    tmp_path: Path,
) -> None:
    # Arrange
    image = np.zeros((10, 10), dtype=np.uint8)

    for directory_name in ["s10", "s2", "s1"]:
        person_directory = tmp_path / directory_name
        person_directory.mkdir()

        cv2.imwrite(
            str(person_directory / "1.pgm"),
            image,
        )

    dataset = AttFaceDataset(tmp_path)

    # Act
    samples = list(dataset.iter_samples())

    labels = [
        label
        for _, label, _ in samples
    ]

    # Assert
    assert labels == [1, 2, 10]