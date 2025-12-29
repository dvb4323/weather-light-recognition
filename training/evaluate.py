import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

from datasets.bdd_dataset import BDDDataset
from models.multitask_model import MultiTaskModel


def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_weather_classes = len(config['classes']['weather'])
    weather_labels = list(range(num_weather_classes))
    num_time_classes = len(config['classes']['timeofday'])
    time_labels = list(range(num_time_classes))

    device = torch.device(
        config['train']['device'] if torch.cuda.is_available() else "cpu"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_eval_dir = os.path.join("evaluation", timestamp)
    report_dir = os.path.join(base_eval_dir, "reports")
    figure_dir = os.path.join(base_eval_dir, "figures")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    val_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])

    test_dataset = BDDDataset(
        img_dir=config['data']['test_images'],  # using val set
        ann_dir=config['data']['test_anns'],
        transforms=val_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    model = MultiTaskModel(
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)

    checkpoint_path = os.path.join(
        config['train']['checkpoint_dir'],
        "best_model.pth"
    )

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

    weather_report = classification_report(
        weather_targets,
        weather_preds,
        labels=weather_labels,
        target_names=config['classes']['weather'],
        zero_division=0
    )

    print("\n--- Weather Classification Report ---")
    print(weather_report)

    with open(os.path.join(report_dir, "weather_report.txt"), "w") as f:
        f.write(weather_report)

    cm_weather = confusion_matrix(
        weather_targets,
        weather_preds,
        labels=weather_labels
    )

    plot_confusion_matrix(
        cm_weather,
        config['classes']['weather'],
        "Weather Confusion Matrix",
        os.path.join(figure_dir, "weather_cm.png")
    )

    time_report = classification_report(
        time_targets,
        time_preds,
        labels=time_labels,
        target_names=config['classes']['timeofday'],
        zero_division=0
    )

    print("\n--- Time of Day Classification Report ---")
    print(time_report)

    with open(os.path.join(report_dir, "time_report.txt"), "w") as f:
        f.write(time_report)

    cm_time = confusion_matrix(
        time_targets,
        time_preds,
        labels=time_labels
    )

    plot_confusion_matrix(
        cm_time,
        config['classes']['timeofday'],
        "Time of Day Confusion Matrix",
        os.path.join(figure_dir, "time_cm.png")
    )

    weather_acc = np.mean(np.array(weather_preds) == np.array(weather_targets))
    time_acc = np.mean(np.array(time_preds) == np.array(time_targets))

    summary = f"""
Evaluation Summary
==================
Timestamp: {timestamp}
Checkpoint: {checkpoint_path}

Weather Accuracy: {weather_acc:.4f}
Time of Day Accuracy: {time_acc:.4f}
"""

    print(summary)

    with open(os.path.join(report_dir, "summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    evaluate()
