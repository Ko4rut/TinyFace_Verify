"""Run a repeatable ATT Faces evaluation for one PCA owner model."""

import argparse
from pathlib import Path

from src.eigenface.preprocessor import FacePreprocessor
from src.evaluation.att_faces import AttFacesBenchmark


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Eigenface verification on ATT Faces.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/att_faces"),
    )
    parser.add_argument("--owner-label", type=int, default=1)
    parser.add_argument("--enrollment-samples", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    benchmark = AttFacesBenchmark(
        dataset_root=arguments.dataset_root,
        preprocessor=FacePreprocessor(),
        n_components=arguments.n_components,
        enrollment_samples=arguments.enrollment_samples,
    )
    evaluation = benchmark.evaluate_owner(arguments.owner_label)
    threshold = evaluation.recommended_threshold

    print(f"Owner label: {arguments.owner_label}")
    print(f"Genuine samples: {evaluation.genuine_sample_count}")
    print(f"Impostor samples: {evaluation.impostor_sample_count}")
    print(f"Recommended threshold: {threshold.threshold:.6f}")
    print(f"False accept rate: {threshold.false_accept_rate:.2%}")
    print(f"False reject rate: {threshold.false_reject_rate:.2%}")
    print(f"Balanced error rate: {threshold.balanced_error_rate:.2%}")


if __name__ == "__main__":
    main()
