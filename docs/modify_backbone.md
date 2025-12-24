Trường hợp 1: Chỉ thay backbone (ResNet → EfficientNet / ViT)
Bạn CẦN làm gì?
✅ (1) Cập nhật models/backbone.py

Thêm class backbone mới

Đảm bảo output là feature vector

Ví dụ:

class EfficientNetBackbone(nn.Module):
    def __init__(self):
        ...
        self.out_dim = 1280

⚠️ (2) Cập nhật heads.py (nếu cần)

Nếu backbone mới có out_dim khác:

WeatherHead(in_dim=backbone.out_dim)


📌 Chỉ là tham số, không đổi logic.

❌ (3) KHÔNG cần sửa

bdd_dataset.py

multitask_model.py

training loop

✅ (4) Chỉnh config.yaml
model:
  backbone: efficientnet_b0

4️⃣ Trường hợp 2: So sánh 2 backbone trong cùng bài

👉 Đây là case bạn chắc chắn sẽ làm.

Cách làm đúng:

Viết 2 backbone class

Chọn bằng config

def build_backbone(name):
    if name == "resnet18":
        return ResNetBackbone()
    elif name == "efficientnet_b0":
        return EfficientNetBackbone()


📌 Training code không đổi.

5️⃣ Trường hợp 3: Thử mô hình khác nhưng vẫn là classification

Ví dụ:

Thay optimizer

Thay loss weight

Thêm dropout

Bạn CHỈ cần:

config.yaml

hoặc heads.py

❌ Không đụng dataset & backbone logic.

6️⃣ Trường hợp 4 (KHÔNG nên với bài này):

Đổi sang bài toán khác

Ví dụ:

Detection

Segmentation

👉 Khi đó:

Phải viết lại dataset

Phải viết lại head

Phải viết lại metric

❌ Không nên, vượt scope.

7️⃣ Bảng tóm tắt: “đụng file nào?”
Thay đổi	File cần sửa
Backbone khác	models/backbone.py, config.yaml
Thêm backbone mới	models/backbone.py
Đổi số class	heads.py, config.yaml
Đổi optimizer	train.py hoặc config.yaml
Đổi dataset	bdd_dataset.py
So sánh mô hình	KHÔNG sửa training logic