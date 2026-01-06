import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class BDDDataset(Dataset):
    """
    BDD100K Dataset for Multi-Task Classification (Weather & Time-of-Day).
    """
    def __init__(self, img_dir, ann_dir, transforms=None, weather_classes=None, time_classes=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.weather_classes = weather_classes if weather_classes else []
        self.time_classes = time_classes if time_classes else []
        
        # Filter image names that have corresponding annotations
        all_imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_names = []
        for img_name in all_imgs:
            if os.path.exists(os.path.join(ann_dir, img_name + '.json')):
                self.img_names.append(img_name)
        
        # Map classes to indices
        self.weather_to_idx = {name: i for i, name in enumerate(self.weather_classes)}
        self.time_to_idx = {name: i for i, name in enumerate(self.time_classes)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name + '.json')
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load annotation
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            
        tags = ann.get('tags', [])
        weather = None
        timeofday = None
        
        for tag in tags:
            if tag['name'] == 'weather':
                weather = tag['value']
                # Map foggy to overcast (foggy class removed - only 110 samples)
                if weather == 'foggy':
                    weather = 'overcast'
            elif tag['name'] == 'timeofday':
                timeofday = tag['value']
        
        # Map to class indices, using 'undefined' for missing/unknown values
        if weather in self.weather_to_idx:
            weather_idx = self.weather_to_idx[weather]
        elif 'undefined' in self.weather_to_idx:
            weather_idx = self.weather_to_idx['undefined']
        else:
            weather_idx = 0  # Fallback
            
        if timeofday in self.time_to_idx:
            time_idx = self.time_to_idx[timeofday]
        elif 'undefined' in self.time_to_idx:
            time_idx = self.time_to_idx['undefined']
        else:
            time_idx = 0  # Fallback
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, {
            'weather': torch.tensor(weather_idx, dtype=torch.long),
            'timeofday': torch.tensor(time_idx, dtype=torch.long)
        }
