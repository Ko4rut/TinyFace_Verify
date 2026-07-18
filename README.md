
# TinyFace Verify

**Lightweight On-Device Face Verification Using Eigenfaces**

TinyFace Verify là hệ thống xác minh khuôn mặt 1:1 sử dụng phương pháp
Eigenfaces/PCA, được thiết kế để chạy cục bộ trên các thiết bị biên có tài
nguyên phần cứng hạn chế.

Project tập trung vào việc tìm hiểu toàn bộ pipeline của một hệ thống sinh
trắc học: đăng ký người dùng, xử lý ảnh khuôn mặt, trích xuất đặc trưng,
so khớp, lựa chọn ngưỡng và đánh giá hiệu năng.

> Đây là project phục vụ học tập và nghiên cứu. Hệ thống không được thiết kế
> để sử dụng như một giải pháp bảo mật trong môi trường thực tế.

---

## 1. Bài toán

Hệ thống giải quyết bài toán **face verification 1:1**:

> Ảnh khuôn mặt hiện tại có thuộc cùng một người với khuôn mặt đã đăng ký
> hay không?

Đầu vào:

- Ảnh hoặc video từ camera.
- Biometric template của người đã đăng ký.

Đầu ra:

- Similarity score hoặc distance score.
- Quyết định `VERIFIED`, `REJECTED` hoặc `UNCERTAIN`.
- Thời gian xử lý của một lần xác minh.

Ví dụ:

```text
Similarity score: 0.84
Threshold:        0.72
Decision:         VERIFIED
Processing time:  42 ms
```
