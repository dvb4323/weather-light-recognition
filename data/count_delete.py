import os
import json
from collections import Counter
import concurrent.futures
import time

# --- CẤU HÌNH ĐIỀU KIỆN XÓA ---
# Các giá trị của weather cần xóa
BAD_WEATHER = {'foggy', 'undefined'}
# Các giá trị của timeofday cần xóa
BAD_TIMEOFDAY = {'undefined'}

def process_and_clean(json_path):
    """
    Hàm này đọc file JSON:
    - Kiểm tra weather và timeofday.
    - Nếu vi phạm điều kiện xóa -> Xóa JSON + Ảnh -> Trả về 'DELETED'
    - Nếu hợp lệ -> Trả về giá trị để đếm
    """
    w_val = None
    t_val = None
    is_deleted = False
    
    try:
        # 1. Đọc nội dung JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tags = data.get('tags', [])
            
            for tag in tags:
                name = tag.get('name')
                if name == 'weather':
                    w_val = tag.get('value')
                elif name == 'timeofday':
                    t_val = tag.get('value')

        # 2. Kiểm tra điều kiện xóa (LOGIC MỚI)
        # Xóa nếu weather xấu HOẶC timeofday xấu
        if (w_val in BAD_WEATHER) or (t_val in BAD_TIMEOFDAY):
            is_deleted = True
            
            # --- Xóa file JSON ---
            os.remove(json_path)
            
            # --- Xử lý đường dẫn để xóa file Ảnh ---
            head, tail = os.path.split(json_path)
            
            # Xử lý đường dẫn thư mục ảnh
            if head.endswith('ann') or head.endswith(os.sep + 'ann'):
                img_dir = head.replace('ann', 'img') 
            else:
                img_dir = head.replace('ann', 'img') 

            # Xử lý tên file ảnh
            file_name = tail.replace('.json', '')
            if not file_name.lower().endswith('.jpg'):
                file_name += '.jpg'
            
            img_path = os.path.join(img_dir, file_name)
            
            # --- Xóa file Ảnh ---
            if os.path.exists(img_path):
                os.remove(img_path)
            
    except Exception as e:
        # print(f"Lỗi xử lý {json_path}: {e}")
        pass

    if is_deleted:
        return 'DELETED', None, None
    else:
        return 'KEPT', w_val, t_val

def filter_and_count():
    total_start_time = time.time()
    splits = ['train', 'val']
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        for split in splits:
            print(f"\n{'='*50}")
            print(f"  ĐANG XỬ LÝ VÀ LÀM SẠCH TẬP: {split.upper()}")
            print(f"{'='*50}")

            ann_dir = os.path.join(split, 'ann')
            
            if not os.path.exists(ann_dir):
                print(f"⚠️ Không tìm thấy thư mục: {ann_dir}")
                continue

            # Quét danh sách file
            print(f"-> Đang quét file trong {ann_dir}...")
            files = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith('.json')]
            total_files = len(files)
            print(f"-> Tìm thấy {total_files} file. Bắt đầu lọc và đếm...")

            if total_files == 0:
                continue

            weather_counter = Counter()
            timeofday_counter = Counter()
            deleted_count = 0
            
            # Xử lý đa luồng
            split_start_time = time.time()
            results = executor.map(process_and_clean, files)

            # Tổng hợp kết quả
            for status, w, t in results:
                if status == 'DELETED':
                    deleted_count += 1
                else:
                    if w: weather_counter[w] += 1
                    if t: timeofday_counter[t] += 1
            
            # --- In báo cáo ---
            duration = time.time() - split_start_time
            remaining_files = total_files - deleted_count
            
            print(f"\n--- KẾT QUẢ {split.upper()} (Xử lý trong {duration:.2f}s) ---")
            print(f"🚫 Đã xóa: {deleted_count} file")
            print(f"   (Lý do: Weather='foggy'/'undefined' HOẶC Timeofday='undefined')")
            print(f"✅ Còn lại: {remaining_files} file")
            
            print(f"\n[Thống kê Weather còn lại - {split.upper()}]")
            if weather_counter:
                for value, count in weather_counter.most_common():
                    print(f"  - {value}: {count}")
            else:
                print("  (Không có dữ liệu)")

            print(f"\n[Thống kê Timeofday còn lại - {split.upper()}]")
            if timeofday_counter:
                for value, count in timeofday_counter.most_common():
                    print(f"  - {value}: {count}")
            else:
                print("  (Không có dữ liệu)")

    print(f"\n{'='*50}")
    print(f"HOÀN TẤT TOÀN BỘ QUÁ TRÌNH TRONG: {time.time() - total_start_time:.2f} giây")

if __name__ == "__main__":
    print("⚠️  CẢNH BÁO: Script này sẽ XÓA VĨNH VIỄN các file thỏa mãn điều kiện sau:")
    print("   1. Weather là 'foggy' hoặc 'undefined'")
    print("   2. Timeofday là 'undefined'")
    confirm = input("Bạn có chắc chắn muốn tiếp tục? (y/n): ")
    if confirm.lower() == 'y':
        filter_and_count()
    else:
        print("Đã hủy.")