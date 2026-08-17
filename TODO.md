# TinyFace Verify Roadmap

This roadmap aims to turn the current project into a measurable and reliable
learning prototype. It does not claim production-grade biometric security.

## Current Baseline

- Runtime: owner-specific PCA with reconstruction MSE over five camera frames.
- Runtime configuration: 20 enrollment frames, 9 PCA components, and a manual
  error threshold in `src/main.py`.
- Offline evaluation: ATT Faces split with five enrollment images, five genuine
  holdout images, 390 impostor images, and four PCA components.
- Example `s1` result: FAR 0.77 percent and FRR 0.00 percent.

The runtime and evaluation configurations are intentionally different today, so
the evaluation threshold must not be copied into the camera application.

## P0: Make the Prototype Correct and Repeatable

### Replace hard-coded startup configuration

- [ ] Add an explicit CLI or configuration object for mode, camera index, model
      path, sample counts, PCA components, and threshold.
- [ ] Remove the duplicate `system_mode` assignment from `src/main.py`.
- [ ] Guard application startup with `if __name__ == "__main__"` so imports do
      not open a camera.

Done when `python -m src.main --mode enrollment` and
`python -m src.main --mode verify` are both explicit, testable commands.

### Repair or remove the unused training path

- [ ] Repair `src/eigenface/training_data_loader.py`, which imports the missing
      `DatasetPreprocessor` class.
- [ ] Replace the current `DataLoader` function and print-only train script with
      a documented, executable training command, or remove them if owner-only
      enrollment is the intended design.
- [ ] Remove or merge the duplicate `src/eigenfaces/` and `src/eigenface/`
      directories so one implementation is canonical.

Done when every Python module can be imported and each documented command runs.

### Store templates safely

- [ ] Replace pickle persistence with `.npz` and load it with
      `allow_pickle=False`.
- [ ] Validate required keys, dtypes, dimensions, PCA component count, and
      finite values before accepting a template.
- [ ] Keep generated templates out of Git and document the local storage path.

Done when a malformed or unexpected template is rejected without executing code.

### Expand automated coverage

- [ ] Add tests for PCA fit/transform/inverse transform and invalid inputs.
- [ ] Add enrollment-to-save-to-load-to-verify integration tests.
- [ ] Add tests for alignment failures, threshold boundary behavior, and template
      schema validation.
- [ ] Add a `pytest.ini` configuration so test discovery and temporary files are
      isolated from workspace artifacts.

Done when the full test suite runs with one documented command in a clean clone.

### Make dependency installation reproducible

- [ ] Pin compatible dependency versions and the supported Python version.
- [ ] Keep only one OpenCV distribution; `opencv-contrib-python` already includes
      the core OpenCV modules used by this project.
- [ ] Remove standard-library and unused packages from `requirements.txt`.

Done when a clean virtual environment installs one deterministic dependency set.

## P1: Make Evaluation Trustworthy

### Evaluate all owners, not one selected owner

- [ ] Add `--all-owners` to the evaluation CLI.
- [ ] Report per-owner, macro-average, median, and worst-case FAR/FRR/BER.
- [ ] Persist a CSV or JSON report containing the configuration, dataset split,
      threshold, and metrics.

Done when a result can answer whether `s1` is representative or an outlier.

### Calibrate thresholds by risk target

- [ ] Add a threshold sweep that reports ROC/DET data and EER.
- [ ] Select thresholds at explicit FAR targets, such as 1 percent, 0.1 percent,
      or a task-specific target, rather than only equalizing FAR and FRR.
- [ ] Create separate calibration and final test splits; never choose a threshold
      and report its final score on the same samples.

Done when the runtime threshold comes from a recorded calibration experiment with
the same preprocessing, PCA configuration, and collection policy.

### Test realistic capture variation

- [ ] Add a local evaluation protocol with separate enrollment and verification
      sessions for each participant.
- [ ] Record lighting, pose, expression, glasses, mask, distance, and camera
      changes where possible.
- [ ] Report results by condition, especially illumination and pose.

Done when performance degradation under the expected operating conditions is
measured rather than assumed.

## P2: Replace Naive Five-Frame Averaging

Five frames are a reasonable latency-oriented starting point, but the current
five frames are sampled 150 ms apart and are usually near-duplicates. The
current mean reconstruction error is therefore not a robust video template.

### Add track continuity and a quality pool

- [ ] Associate detections across frames and collect only one stable face track.
- [ ] Capture a short time window, such as 1 to 2 seconds, into a candidate pool
      instead of accepting the first five eligible frames.
- [ ] Score candidates by detection confidence, sharpness, face size, pose,
      exposure, and diversity from already selected samples.
- [ ] Retain the best diverse 3 to 5 frames rather than five adjacent frames.

Done when all selected samples belong to one track and the UI can explain why a
frame was retained or skipped.

### Use a robust session decision

- [ ] Compare `mean`, `median`, trimmed mean, and an "at least M of N frames
      pass" rule in offline evaluation.
- [ ] Choose the decision rule from FAR/FRR results, not intuition.
- [ ] Record per-frame errors and summary statistics in debug output.

Done when a single blurry, occluded, or swapped frame cannot silently dominate a
verification result.

## P3: Improve Recognition and Security

### Improve illumination handling

- [ ] Add exposure and brightness-asymmetry checks before enrollment and verify.
- [ ] Compare CLAHE-only preprocessing with gamma or Retinex normalization using
      the same evaluation protocol.
- [ ] Require enrollment samples from multiple lighting conditions.

Done when the illumination trade-off is measured in FAR/FRR, not inferred from a
single threshold change.

### Add presentation-attack protection

- [ ] Define the threat model: printed photo, replayed video, screen image, or
      physical mask.
- [ ] Add an active challenge or a passive presentation-attack detector.
- [ ] Measure false accepts on presentation attacks separately from ordinary
      impostor tests.

Done when the system can state which attacks it detects and its measured failure
rate for each one.

### Decide whether PCA remains the model

- [ ] Build an experiment that compares the current owner-PCA/MSE approach with
      a global-PCA/cosine baseline.
- [ ] If practical accuracy is required, compare a lightweight pretrained face
      embedding model, such as ArcFace or MobileFaceNet, using the same
      enrollment, verification, and evaluation protocol.
- [ ] Keep the model only if it meets the agreed FAR, FRR, latency, memory, and
      privacy targets.

Done when model selection is based on a reproducible comparison rather than a
single-owner benchmark.

## Technical Debt Register

| Priority | Issue | Current Location | Resolution |
| --- | --- | --- | --- |
| P0 | Startup mode is overwritten and camera starts on import. | `src/main.py` | Introduce CLI/configuration and a guarded `main()` function. |
| P0 | Unused training path cannot import. | `src/eigenface/training_data_loader.py` | Repair or remove it. |
| P0 | Templates use unsafe pickle serialization. | `src/enrollment/`, `src/verification/` | Replace with validated `.npz`. |
| P0 | Dependencies are unpinned and include overlapping OpenCV packages. | `requirements.txt` | Pin supported versions and select one OpenCV package. |
| P1 | Runtime threshold is manually copied from a mismatched evaluation setup. | `src/main.py` | Calibrate using an identical runtime profile. |
| P1 | Evaluation covers one selected owner and one fixed split. | `src/evaluation/` | Add all-owner and cross-validation reporting. |
| P2 | Five adjacent frames have no tracking, diversity, or quality-weighted fusion. | `src/capture/`, `src/camera/` | Add a stable track and quality-based candidate selection. |
| P2 | The verification model is loaded from disk for every completed session. | `src/verification/` | Cache a validated immutable template and provide explicit reload behavior. |
| P3 | No presentation-attack defense or biometric data lifecycle policy. | Project-wide | Add liveness, consent, retention, deletion, and protected storage rules. |
