
| Thành phần                  | Trách nhiệm                                              |
| ----------------------------- | ---------------------------------------------------------- |
| `EigenfacePreprocessor`     | Grayscale, normalize, flatten ảnh aligned                 |
| `EigenfaceTrainer`          | Học PCA từ dataset và tạo mô hình Eigenfaces         |
| `EigenfaceFeatureExtractor` | Dùng PCA đã train để biến ảnh thành embedding      |
| `EigenfaceModelRepository`  | Lưu/load PCA bằng`joblib`                              |
| `FaceVerificationService`   | Tạo template trung bình và so sánh hai khuôn mặt     |
| `CameraRuntime`             | Điều phối camera, thu đủ frame rồi gọi verification |
