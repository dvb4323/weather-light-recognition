# Weather & Time-of-Day Multi-Task Classification

Dự án nhận diện điều kiện môi trường (Thời tiết và Thời gian trong ngày) từ ảnh camera sử dụng bộ dữ liệu BDD100K.

## 🚀 Tính năng
- **Multi-task Learning**: Sử dụng 1 backbone chung (ResNet18) và 2 heads phân loại riêng biệt.
- **Dễ cấu hình**: Điều chỉnh tham số qua `config.yaml`.
- **Đầy đủ Pipeline**: Từ tiền xử lý dữ liệu, huấn luyện, đánh giá đến suy diễn (inference).

## 📁 Cấu trúc thư mục
```
project_root/
├── data/               # Chứa dữ liệu train/val/test
├── datasets/
│   └── bdd_dataset.py  # Loader cho dữ liệu BDD100K
├── models/
│   ├── backbone.py     # Shared backbone (ResNet)
│   ├── heads.py        # Classification heads
│   └── multitask_model.py
├── training/
│   ├── train.py        # Huấn luyện mô hình
│   └── evaluate.py     # Đánh giá (Accuracy, F1, Confusion Matrix)
├── inference/
│   └── infer.py        # Suy diễn trên 1 ảnh
├── utils/
│   ├── metrics.py      # Các hàm đo lường
│   └── visualization.py # Trực quan hóa kết quả
├── config.yaml         # Cấu hình chính
└── README.md
```

## 🛠 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt dữ liệu vào thư mục `data/` theo cấu trúc:

- `data/train/img` & `data/train/ann`

- `data/val/img` & `data/val/ann`

### 2. Huấn luyện

```bash
python -m training.train
```
Checkpoints sẽ được lưu trong thư mục `checkpoints/`.

### 3. Đánh giá

```bash
python -m training.evaluate
```
Lệnh này sẽ tạo ra Confusion Matrix và in báo cáo F1-score.

### 4. Suy diễn (Inference)

```bash
python -m inference.infer --image path/to/image.jpg --model checkpoints/best_model.pth
```

## 📊 Kết quả mong đợi

Mô hình sẽ xuất ra dự đoán dưới dạng:
```json
{
  "weather": "rainy",
  "timeofday": "night"
}
```

## 📝 Yêu cầu hệ thống

- PyTorch, Torchvision
- PIL, NumPy, YAML
- Scikit-learn, Matplotlib, Seaborn (cho evaluation)
