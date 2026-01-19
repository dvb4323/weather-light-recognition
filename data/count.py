import os
import json
from collections import Counter
import concurrent.futures
import time

# --- Hàm xử lý cho 1 file (Giữ nguyên) ---
def process_single_file(file_path):
    w_val = None
    t_val = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tags = data.get('tags', [])
            
            for tag in tags:
                name = tag.get('name')
                if name == 'weather':
                    w_val = tag.get('value')
                elif name == 'timeofday':
                    t_val = tag.get('value')
    except Exception:
        pass 
    return w_val, t_val

def count_attributes_separated():
    total_start_time = time.time()
    splits = ['train', 'val']
    
    # Khởi tạo Executor một lần để tái sử dụng thread pool (tối ưu tài nguyên)
    # max_workers: Tự động điều chỉnh dựa trên số core CPU
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        for split in splits:
            print(f"\n{'='*40}")
            print(f"  ĐANG XỬ LÝ TẬP: {split.upper()}")
            print(f"{'='*40}")

            ann_dir = os.path.join(split, 'ann')
            
            # 1. Kiểm tra thư mục
            if not os.path.exists(ann_dir):
                print(f"⚠️ Không tìm thấy thư mục: {ann_dir}")
                continue

            # 2. Quét file trong thư mục hiện tại
            print(f"-> Đang quét file trong {ann_dir}...")
            files = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith('.json')]
            num_files = len(files)
            print(f"-> Tìm thấy {num_files} file. Đang đếm...")

            if num_files == 0:
                continue

            # 3. Reset bộ đếm cho riêng tập này
            weather_counter = Counter()
            timeofday_counter = Counter()
            
            # 4. Xử lý đa luồng
            split_start_time = time.time()
            results = executor.map(process_single_file, files)

            # 5. Tổng hợp kết quả
            for w, t in results:
                if w: weather_counter[w] += 1
                if t: timeofday_counter[t] += 1
            
            # 6. In kết quả của tập hiện tại
            print(f"\n--- THỐNG KÊ {split.upper()} (Xử lý trong {time.time() - split_start_time:.2f}s) ---")
            
            print(f"\n[Weather - {split.upper()}]")
            if weather_counter:
                for value, count in weather_counter.most_common():
                    print(f"  - {value}: {count}")
            else:
                print("  (Không có dữ liệu)")

            print(f"\n[Timeofday - {split.upper()}]")
            if timeofday_counter:
                for value, count in timeofday_counter.most_common():
                    print(f"  - {value}: {count}")
            else:
                print("  (Không có dữ liệu)")

    print(f"\n{'='*40}")
    print(f"TỔNG THỜI GIAN HOÀN THÀNH: {time.time() - total_start_time:.2f} giây")

if __name__ == "__main__":
    count_attributes_separated()