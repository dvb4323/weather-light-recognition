import os
import json
import shutil
import random
from collections import defaultdict, Counter
import concurrent.futures

# --- CẤU HÌNH ---
SOURCE_DIR = 'train'       # Thư mục nguồn
TARGET_DIR = 'test'        # Thư mục đích
MIN_SAMPLES = 700          # Số lượng tối thiểu mỗi thuộc tính
TEST_RATIO = 0.1           # Tỉ lệ tập test mong muốn (10%)

# Các thuộc tính cần quan tâm
ATTR_KEYS = ['weather', 'timeofday']

def read_file_metadata(json_path):
    """Đọc file JSON và trả về metadata cần thiết"""
    meta = {'weather': None, 'timeofday': None}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tags = data.get('tags', [])
            for tag in tags:
                name = tag.get('name')
                if name in ATTR_KEYS:
                    meta[name] = tag.get('value')
        return json_path, meta
    except Exception:
        return json_path, None

def split_dataset():
    # --- ĐỊNH NGHĨA ĐƯỜNG DẪN NGUỒN ---
    ann_src = os.path.join(SOURCE_DIR, 'ann')
    img_src = os.path.join(SOURCE_DIR, 'img')
    
    # --- ĐỊNH NGHĨA ĐƯỜNG DẪN ĐÍCH (Đảm bảo có ann và img trong test) ---
    ann_dst = os.path.join(TARGET_DIR, 'ann') # -> test/ann
    img_dst = os.path.join(TARGET_DIR, 'img') # -> test/img

    # 1. Kiểm tra dữ liệu nguồn
    if not os.path.exists(ann_src):
        print(f"Lỗi: Không tìm thấy thư mục nguồn {ann_src}")
        return

    # 2. Tạo cấu trúc thư mục cho tập TEST
    print(f"-> Đang tạo cấu trúc thư mục:")
    print(f"   + {ann_dst}")
    print(f"   + {img_dst}")
    os.makedirs(ann_dst, exist_ok=True)
    os.makedirs(img_dst, exist_ok=True)

    # 3. Quét toàn bộ metadata từ tập TRAIN
    print("-> Đang đọc dữ liệu từ tập Train...")
    files = [os.path.join(ann_src, f) for f in os.listdir(ann_src) if f.endswith('.json')]
    
    file_db = {} 
    attr_index = defaultdict(list) 

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(read_file_metadata, files)
        for fpath, meta in results:
            if meta:
                file_db[fpath] = meta
                if meta['weather']:
                    attr_index[f'weather:{meta["weather"]}'].append(fpath)
                if meta['timeofday']:
                    attr_index[f'timeofday:{meta["timeofday"]}'].append(fpath)

    total_files = len(file_db)
    target_total_size = int(total_files * TEST_RATIO)
    print(f"   Tổng file train hiện có: {total_files}")
    print(f"   Mục tiêu tập test: ~{target_total_size} file (10%)")

    # 4. Thuật toán chọn file (Greedy Selection)
    selected_files = set()
    current_counts = Counter()

    all_attr_values = list(attr_index.keys())
    
    print("\n-> Đang lựa chọn ảnh (ưu tiên đủ 700 ảnh/thuộc tính)...")
    
    # 4a. Lấp đầy tối thiểu 700
    while True:
        most_needed_attr = None
        max_missing = 0
        
        for attr_key in all_attr_values:
            current_val = current_counts[attr_key]
            if current_val < MIN_SAMPLES:
                missing = MIN_SAMPLES - current_val
                available_candidates = [f for f in attr_index[attr_key] if f not in selected_files]
                if not available_candidates:
                    continue 
                
                if missing > max_missing:
                    max_missing = missing
                    most_needed_attr = attr_key

        if most_needed_attr is None:
            break 

        candidates = [f for f in attr_index[most_needed_attr] if f not in selected_files]
        picked_file = random.choice(candidates)
        selected_files.add(picked_file)
        
        meta = file_db[picked_file]
        if meta['weather']: current_counts[f'weather:{meta["weather"]}'] += 1
        if meta['timeofday']: current_counts[f'timeofday:{meta["timeofday"]}'] += 1

    print(f"   Đã chọn {len(selected_files)} file để thỏa mãn điều kiện min={MIN_SAMPLES}.")

    # 4b. Lấp đầy cho đủ 10%
    if len(selected_files) < target_total_size:
        remaining_needed = target_total_size - len(selected_files)
        print(f"   Đang lấy thêm {remaining_needed} file ngẫu nhiên để đủ 10%...")
        
        all_files_list = list(file_db.keys())
        available_files = [f for f in all_files_list if f not in selected_files]
        
        if len(available_files) >= remaining_needed:
            extras = random.sample(available_files, remaining_needed)
            selected_files.update(extras)
            for f in extras:
                meta = file_db[f]
                if meta['weather']: current_counts[f'weather:{meta["weather"]}'] += 1
                if meta['timeofday']: current_counts[f'timeofday:{meta["timeofday"]}'] += 1
        else:
            selected_files.update(available_files)

    # 5. Thực hiện di chuyển file
    print(f"\n-> Bắt đầu di chuyển {len(selected_files)} file sang '{TARGET_DIR}'...")
    
    moved_count = 0
    for json_path in selected_files:
        try:
            filename_json = os.path.basename(json_path)
            
            # Xác định tên ảnh
            file_name_core = filename_json.replace('.json', '')
            if not file_name_core.lower().endswith('.jpg'):
                file_name_core += '.jpg'
            
            src_img_path = os.path.join(img_src, file_name_core)
            
            # --- ĐƯỜNG DẪN ĐÍCH CỤ THỂ ---
            # File json vào test/ann
            dst_json_path = os.path.join(ann_dst, filename_json)
            # File ảnh vào test/img
            dst_img_path = os.path.join(img_dst, file_name_core)
            
            # Di chuyển JSON
            shutil.move(json_path, dst_json_path)
            
            # Di chuyển Ảnh (nếu có)
            if os.path.exists(src_img_path):
                shutil.move(src_img_path, dst_img_path)
            
            moved_count += 1
            
        except Exception as e:
            print(f"Lỗi khi di chuyển {json_path}: {e}")

    # 6. In báo cáo
    print("\n" + "="*40)
    print("HOÀN TẤT! THỐNG KÊ TẬP TEST:")
    print(f"Tổng số file: {moved_count}")
    print("="*40)
    
    sorted_stats = sorted(current_counts.items(), key=lambda x: x[0])
    
    print("\n[Chi tiết số lượng từng thuộc tính]")
    for attr, count in sorted_stats:
        status = "✅ Đạt" if count >= MIN_SAMPLES else "⚠️ Thiếu (Do gốc không đủ)"
        print(f"  - {attr:<25} : {count:5d} ({status})")

if __name__ == "__main__":
    print(f"⚠️  CHÚ Ý: Script này sẽ lấy ~10% ảnh từ '{SOURCE_DIR}' chuyển sang '{TARGET_DIR}'.")
    print(f"    Cấu trúc tạo ra: {TARGET_DIR}/ann và {TARGET_DIR}/img")
    confirm = input("Bạn có muốn tiếp tục? (y/n): ")
    if confirm.lower() == 'y':
        split_dataset()
    else:
        print("Đã hủy.")