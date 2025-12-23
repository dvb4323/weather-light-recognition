import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from datasets.bdd_dataset import BDDDataset
from models.multitask_model import MultiTaskModel

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def evaluate():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

    # Transforms
    val_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    # Dataset
    test_dataset = BDDDataset(
        img_dir=config['data']['val_images'], # Using val for evaluation example
        ann_dir=config['data']['val_anns'],
        transforms=val_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)

    # Model
    model = MultiTaskModel(
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)
    
    checkpoint_path = os.path.join(config['train']['checkpoint_dir'], "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Evaluating with random weights.")

    model.eval()
    
    weather_preds, weather_targets = [], []
    time_preds, time_targets = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            
            w_pred = outputs['weather'].argmax(dim=1).cpu().numpy()
            t_pred = outputs['timeofday'].argmax(dim=1).cpu().numpy()
            
            weather_preds.extend(w_pred)
            weather_targets.extend(targets['weather'].numpy())
            time_preds.extend(t_pred)
            time_targets.extend(targets['timeofday'].numpy())

    # Metrics - Weather
    print("\n--- Weather Classification Report ---")
    print(classification_report(weather_targets, weather_preds, target_names=config['classes']['weather']))
    
    cm_weather = confusion_matrix(weather_targets, weather_preds)
    plot_confusion_matrix(cm_weather, config['classes']['weather'], "Weather Confusion Matrix", "weather_cm.png")

    # Metrics - Time of Day
    print("\n--- Time of Day Classification Report ---")
    print(classification_report(time_targets, time_preds, target_names=config['classes']['timeofday']))
    
    cm_time = confusion_matrix(time_targets, time_preds)
    plot_confusion_matrix(cm_time, config['classes']['timeofday'], "Time of Day Confusion Matrix", "time_cm.png")

if __name__ == "__main__":
    import os
    evaluate()
