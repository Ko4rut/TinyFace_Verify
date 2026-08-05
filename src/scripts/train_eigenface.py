
from src.eigenface.dataset import AttFaceDataset 
from pathlib import Path
import cv2

root_dir = Path("./data/raw/att_faces") 

# 2. Khởi tạo Dataset
dataset = AttFaceDataset(root_dir)

# 3. Duyệt qua từng mẫu ảnh và hiển thị
for image, label, a in dataset.iter_samples():
    cv2.imshow("AT&T Face Sample", image)
    
    # Nhấn phím bất kỳ để xem ảnh tiếp theo, nhấn 'q' để thoát
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()