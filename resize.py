import os
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# Cấu hình đường dẫn (Sửa lại cho đúng thư mục của bạn)
SRC_TRAIN = 'data/train/img'
DST_TRAIN = 'data/train_224/img'
SRC_VAL = 'data/val/img'
DST_VAL = 'data/val_224/img'
TARGET_SIZE = (224, 224)

def process_image(args):
    src_path, dst_path = args
    try:
        with Image.open(src_path) as img:
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            img.save(dst_path, quality=95)
    except Exception as e:
        print(f"Error {src_path}: {e}")

def run_resize(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        print(f"Not found: {src_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    tasks = [(os.path.join(src_dir, f), os.path.join(dst_dir, f)) for f in files]
    
    print(f"Resizing {len(tasks)} images from {src_dir} to {dst_dir}...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks)))

if __name__ == '__main__':
    run_resize(SRC_TRAIN, DST_TRAIN)
    run_resize(SRC_VAL, DST_VAL)