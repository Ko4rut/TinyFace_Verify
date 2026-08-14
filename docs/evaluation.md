# Eigenface Evaluation

Run an offline benchmark without opening the camera:

```powershell
.\venv\Scripts\python.exe -m src.scripts.evaluate_eigenface --owner-label 1
```

For the selected owner, the benchmark uses the first five ATT Faces images for
enrollment, the remaining five as genuine holdout samples, and all images from
other identities as impostor samples. It sweeps observed reconstruction errors
and reports the threshold where false-accept and false-reject rates are closest.

This measures the PCA method only. ATT Faces images do not go through the
runtime YuNet detection, live-camera quality checks, or liveness detection.
