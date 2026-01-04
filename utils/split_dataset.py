"""
Split the original 70K training set into:
- New train: 60K images
- New val: 10K images
- Test: Use original val set (10K images)

Usage: python utils/split_dataset.py
"""

import os
import shutil
import random
from collections import Counter
import json

def split_train_dataset(
    source_img_dir,
    source_ann_dir,
    output_base_dir,
    train_ratio=0.857,  # 60K / 70K = 0.857
    seed=42
):
    """Split original train set into new train and val sets."""
    
    random.seed(seed)
    
    # Create output directories
    new_train_img = os.path.join(output_base_dir, 'new_train', 'img')
    new_train_ann = os.path.join(output_base_dir, 'new_train', 'ann')
    new_val_img = os.path.join(output_base_dir, 'new_val', 'img')
    new_val_ann = os.path.join(output_base_dir, 'new_val', 'ann')
    
    for dir_path in [new_train_img, new_train_ann, new_val_img, new_val_ann]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    all_imgs = [f for f in os.listdir(source_img_dir) if f.endswith('.jpg')]
    
    # Filter to only those with annotations
    valid_imgs = []
    for img_name in all_imgs:
        ann_path = os.path.join(source_ann_dir, img_name + '.json')
        if os.path.exists(ann_path):
            valid_imgs.append(img_name)
    
    print(f"Total valid images: {len(valid_imgs)}")
    
    # Shuffle and split
    random.shuffle(valid_imgs)
    split_idx = int(len(valid_imgs) * train_ratio)
    
    train_imgs = valid_imgs[:split_idx]
    val_imgs = valid_imgs[split_idx:]
    
    print(f"\nSplit:")
    print(f"  New train: {len(train_imgs)} images (~60K)")
    print(f"  New val:   {len(val_imgs)} images (~10K)")
    
    # Copy files
    def copy_files(img_list, dest_img_dir, dest_ann_dir):
        for img_name in img_list:
            # Copy image
            src_img = os.path.join(source_img_dir, img_name)
            dst_img = os.path.join(dest_img_dir, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Copy annotation
            ann_file = img_name + '.json'
            src_ann = os.path.join(source_ann_dir, ann_file)
            dst_ann = os.path.join(dest_ann_dir, ann_file)
            shutil.copy2(src_ann, dst_ann)
    
    print("\nCopying train files...")
    copy_files(train_imgs, new_train_img, new_train_ann)
    
    print("Copying val files...")
    copy_files(val_imgs, new_val_img, new_val_ann)
    
    print("\n✅ Done!")
    print("\nUpdate config.yaml with:")
    print(f"  train_images: 'data/new_train/img'")
    print(f"  train_anns: 'data/new_train/ann'")
    print(f"  val_images: 'data/new_val/img'")
    print(f"  val_anns: 'data/new_val/ann'")
    print(f"  test_images: 'data/val/img'  # Original val set")
    print(f"  test_anns: 'data/val/ann'")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    source_img = os.path.join(base_dir, 'data', 'train', 'img')
    source_ann = os.path.join(base_dir, 'data', 'train', 'ann')
    output_base = os.path.join(base_dir, 'data')
    
    print("="*60)
    print("Splitting Original 70K Train Set")
    print("="*60)
    print(f"Source: {source_ann}")
    print(f"Target: 60K train + 10K val")
    print("="*60)
    
    split_train_dataset(source_img, source_ann, output_base)
