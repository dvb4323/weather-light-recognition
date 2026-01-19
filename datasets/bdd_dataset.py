import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class BDDDataset(Dataset):
    """
    BDD100K Dataset - Clean Version.
    - Đã loại bỏ logic xử lý 'foggy' và 'undefined' (vì dữ liệu đã sạch).
    - Tích hợp Cache RAM để train siêu nhanh (giảm CPU bottleneck).
    """
    def __init__(self, img_dir, ann_dir, transforms=None, weather_classes=None, time_classes=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.weather_classes = weather_classes if weather_classes else []
        self.time_classes = time_classes if time_classes else []
        
        # Tạo từ điển map tên class sang số index (Ví dụ: 'rainy' -> 2)
        self.weather_to_idx = {name: i for i, name in enumerate(self.weather_classes)}
        self.time_to_idx = {name: i for i, name in enumerate(self.time_classes)}

        # List chứa dữ liệu để train (được nạp vào RAM)
        self.samples = []

        print(f"-> Đang quét và nạp dữ liệu từ: {ann_dir}...")
        
        # Lấy danh sách file JSON
        if os.path.exists(ann_dir):
            json_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        else:
            print(f"Lỗi: Không tìm thấy thư mục {ann_dir}")
            json_files = []

        # --- GIAI ĐOẠN CACHE (Chạy 1 lần duy nhất) ---
        for j_file in tqdm(json_files, desc="Caching Labels"):
            try:
                # 1. Xác định đường dẫn ảnh
                # Logic: file json tên "abc.jpg.json" hoặc "abc.json" -> ảnh "abc.jpg"
                img_name = j_file.replace('.json', '')
                if not img_name.lower().endswith('.jpg'):
                    img_name += '.jpg'
                
                img_path = os.path.join(self.img_dir, img_name)
                
                # Bỏ qua nếu không có file ảnh tương ứng
                if not os.path.exists(img_path):
                    continue

                # 2. Đọc file JSON
                json_path = os.path.join(self.ann_dir, j_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                tags = data.get('tags', [])
                w_val = None
                t_val = None

                # Lấy giá trị tags
                for tag in tags:
                    if tag['name'] == 'weather':
                        w_val = tag['value']
                    elif tag['name'] == 'timeofday':
                        t_val = tag['value']

                # 3. Kiểm tra tính hợp lệ (Logic Sạch)
                # Chỉ lấy file nếu cả weather và timeofday đều nằm trong danh sách class cho phép
                if (w_val in self.weather_to_idx) and (t_val in self.time_to_idx):
                    
                    self.samples.append({
                        'img_path': img_path,
                        'weather_idx': self.weather_to_idx[w_val],
                        'time_idx': self.time_to_idx[t_val]
                    })
                else:
                    # Nếu file lọt lưới (vẫn chứa foggy/undefined hoặc class lạ), tự động bỏ qua
                    # print(f"Bỏ qua file {j_file}: Chứa nhãn không hợp lệ ({w_val}, {t_val})")
                    pass

            except Exception as e:
                continue

        print(f"-> Đã nạp thành công {len(self.samples)} mẫu dữ liệu sạch vào RAM.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Lấy dữ liệu từ RAM (Tốc độ tức thì)
        sample = self.samples[idx]
        
        # 1. Đọc ảnh
        try:
            image = Image.open(sample['img_path']).convert("RGB")
        except Exception:
            # Fallback phòng trường hợp lỗi ổ cứng lúc đọc file
            image = Image.new('RGB', (224, 224))
            
        # 2. Transform
        if self.transforms:
            image = self.transforms(image)
            
        # 3. Trả về nhãn (Đã là số int, không cần map lại)
        return image, {
            'weather': torch.tensor(sample['weather_idx'], dtype=torch.long),
            'timeofday': torch.tensor(sample['time_idx'], dtype=torch.long)
        }