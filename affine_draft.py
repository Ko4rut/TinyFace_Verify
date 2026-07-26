import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 0. CHUẨN BỊ ẢNH GỐC
# -----------------------------------------------------------
# Tạo một bức ảnh (canvas) màu đen kích thước 400x400 pixel
img_size = 400
image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Vẽ một hình chữ nhật màu xanh lá cây nằm ở giữa ảnh
# Điểm góc trên-trái (100, 100), góc dưới-phải (300, 200)
cv2.rectangle(image, (100, 100), (300, 200), (0, 255, 0), -1)

# Hàm tiện ích để chạy cv2.warpAffine
def apply_affine(img, matrix):
    # warpAffine cần 3 tham số: Ảnh đầu vào, Ma trận biến đổi M, (Chiều_rộng_Output, Chiều_cao_Output)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))


# -----------------------------------------------------------
# 1. PHÉP DỊCH CHUYỂN (TRANSLATION)
# -----------------------------------------------------------
# Dịch sang phải 50 pixel (tx=50), dịch xuống dưới 80 pixel (ty=80)
tx, ty = 50, 100
M_translate = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])
img_translated = apply_affine(image, M_translate)


# -----------------------------------------------------------
# 2. PHÉP THU PHÓNG (SCALING)
# -----------------------------------------------------------
# Phóng to theo trục X gấp 1.5 lần, thu nhỏ theo trục Y còn 0.5 lần
sx, sy = 1.5, 0.5
M_scale = np.float32([
    [sx, 0,  0],
    [0,  sy, 0]
])
img_scaled = apply_affine(image, M_scale)


# -----------------------------------------------------------
# 3. PHÉP XOAY (ROTATION)
# -----------------------------------------------------------
# Thay vì tự tính cos(góc) và sin(góc) đưa vào ma trận, 
# OpenCV cung cấp sẵn hàm getRotationMatrix2D cho nhanh.
# Thông số: (Tâm xoay, Góc xoay, Tỷ lệ thu phóng)
center = (img_size // 2, img_size // 2)
angle = 45 # Xoay 45 độ ngược chiều kim đồng hồ
M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
img_rotated = apply_affine(image, M_rotate)


# -----------------------------------------------------------
# 4. PHÉP KÉO NGHIÊNG / XÔ LỆCH (SHEARING)
# -----------------------------------------------------------
# Kéo xô lệch theo trục X (biến hình chữ nhật thành hình bình hành)
shx = 0.5 # Giá trị xô lệch
shy = 0.0
M_shear = np.float32([
    [1,   shx, 0],
    [shy, 1,   0]
])
img_sheared = apply_affine(image, M_shear)


# -----------------------------------------------------------
# HIỂN THỊ KẾT QUẢ BẰNG MATPLOTLIB
# -----------------------------------------------------------
# Chuyển hệ màu từ BGR (của OpenCV) sang RGB (của Matplotlib) để hiển thị đúng màu
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_translated_rgb = cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB)
img_scaled_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
img_rotated_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
img_sheared_rgb = cv2.cvtColor(img_sheared, cv2.COLOR_BGR2RGB)

titles = ['1. Original', '2. Translated (Move)', '3. Scaled', '4. Rotated', '5. Sheared']
images = [image_rgb, img_translated_rgb, img_scaled_rgb, img_rotated_rgb, img_sheared_rgb]

plt.figure(figsize=(15, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off') # Tắt trục toạ độ cho đẹp

plt.tight_layout()
plt.show()