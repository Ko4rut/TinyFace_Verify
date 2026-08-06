import cv2
from pathlib import Path
from src.eigenface.dataset import AttFaceDataset
from src.eigenface.preprocessor import DatasetPreprocessor


root_dir = Path("./data/raw/att_faces") 

# 2. Khởi tạo Dataset
dataset = AttFaceDataset(root_dir)
preprocessor = DatasetPreprocessor()
# 3. Duyệt qua từng mẫu ảnh và hiển thị
for image, label, a in dataset.iter_samples():
    print(image.shape)
    img_processed = preprocessor.preprocess(image)
    print(img_processed)
    cv2.imshow("",img_processed)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()