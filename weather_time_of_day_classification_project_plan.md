# Weather & Time-of-Day Classification for Autonomous Driving

## 1. Mục tiêu dự án

### 1.1 Bối cảnh
Trong hệ thống xe tự hành, điều kiện **thời tiết** và **ánh sáng theo thời gian trong ngày** (time-of-day) ảnh hưởng trực tiếp đến chất lượng nhận thức (perception), đặc biệt là các module như object detection, lane detection. Do đó, việc xây dựng một **module nhận diện điều kiện môi trường** từ ảnh camera là cần thiết để tăng tính bền vững và an toàn cho toàn hệ thống.

### 1.2 Mục tiêu
- Xây dựng mô hình **phân loại ảnh** để:
  - Nhận diện **thời tiết (Weather)**
  - Nhận diện **thời gian trong ngày (Time-of-Day)**
- Áp dụng **Multi-task Learning** với backbone chia sẻ.
- So sánh **baseline CNN** và **mô hình cải tiến (EfficientNet / ViT)**.


---

## 2. Dataset

### 2.1 Dataset sử dụng
- **BDD100K – Image-level Tags**
- Mỗi ảnh có các tag:
  - `weather`
  - `scene`
  - `timeofday`

### 2.2 Các task và class

#### Task 1 – Weather Classification
| Class | Ghi chú |
|------|--------|
| clear | Trời quang |
| partly cloudy | Có mây nhẹ |
| overcast | Trời nhiều mây |
| rainy | Mưa |
| snowy | Tuyết |
| foggy | Sương mù (ít dữ liệu) |

> **Lưu ý**: Có thể cân nhắc gộp `partly cloudy` + `overcast` nếu mất cân bằng quá nặng.

#### Task 2 – Time-of-Day Classification
| Class |
|------|
| daytime |
| dawn/dusk |
| night |



---

## 3. Bài toán & Định nghĩa học máy

### 3.1 Định nghĩa bài toán
Bài toán được mô hình hóa như **Multi-task Image Classification**:

- Input: Ảnh RGB từ camera xe
- Output:
  - Nhãn thời tiết (Weather)
  - Nhãn thời gian trong ngày (Time-of-Day)

### 3.2 Lý do chọn Multi-task Learning
- Hai task chia sẻ đặc trưng thị giác chung (ánh sáng, độ tương phản, màu sắc)
- Giảm chi phí huấn luyện so với hai model riêng biệt
- Phù hợp với kiến trúc perception thực tế

---

## 4. Kiến trúc mô hình

### 4.1 Tổng quan pipeline

```
Image
  ↓
Preprocessing & Augmentation
  ↓
Shared Backbone (CNN / ViT)
  ↓
Global Feature Vector
  ↓              ↓
Weather Head   Time Head
Softmax       Softmax
```

### 4.2 Backbone đề xuất
- **Baseline**: ResNet18 (pretrained ImageNet)
- **Phương pháp so sánh**: EfficientNet-B0 hoặc ViT-Tiny

### 4.3 Classification Heads
- Fully Connected Layer
- Output size:
  - Weather head: N_weather classes
  - Time head: 3 classes

### 4.4 Loss Function

```
L_total = L_weather + L_time
```

- Cross-Entropy Loss cho mỗi head
- Có thể dùng class weights cho Weather

---

## 5. Quy trình thực hiện chi tiết

### 5.1 Data Preprocessing


### 5.2 Data Augmentation
Áp dụng cho tập train:
- Random resize & crop
- Horizontal flip
- Color jitter (brightness, contrast)

Không augmentation cho val/test.

---

## 6. Training Pipeline

### 6.1 Thiết lập huấn luyện
- Framework: PyTorch
- Optimizer: Adam / AdamW
- Learning rate: 1e-4
- Batch size: 32 (tùy GPU)
- Epochs: 15–25

### 6.2 Training loop (pseudo)

```
for epoch in epochs:
  for images, weather_label, time_label in dataloader:
    features = backbone(images)
    weather_pred = weather_head(features)
    time_pred = time_head(features)

    loss = CE(weather_pred, weather_label)
         + CE(time_pred, time_label)

    backprop + update
```

### 6.3 Checkpoint & Logging
- Lưu model tốt nhất theo val loss
- Log:
  - Loss từng task
  - Accuracy từng task

---

## 7. Inference & Evaluation

### 7.1 Inference

```
image → backbone → features
        → weather_head → weather class
        → time_head → time-of-day class
```

### 7.2 Metrics

**Weather**:
- Accuracy
- Macro F1-score
- Confusion Matrix

**Time-of-Day**:
- Accuracy
- F1-score
- Confusion Matrix

