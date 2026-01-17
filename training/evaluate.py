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
import argparse

from datasets.bdd_dataset import BDDDataset
from models.multitask_model import MultiTaskModel
from utils.advanced_metrics import top_k_accuracy, confidence_by_class, average_confidence


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

def evaluate(backbone_name='resnet18'):
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

    # Create evaluation directory with backbone name and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_eval_dir = os.path.join("evaluation", f"{backbone_name}_{timestamp}")
    report_dir = os.path.join(base_eval_dir, "reports")
    figure_dir = os.path.join(base_eval_dir, "figures")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    print(f"Evaluation results will be saved to: {base_eval_dir}")

    val_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])

    # Test dataset (using original val set which has proper labels)
    test_dataset = BDDDataset(
        img_dir=config['data']['test_images'],
        ann_dir=config['data']['test_anns'],
        transforms=val_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'] * 2,  # Larger batch for faster evaluation
        shuffle=False,
        num_workers=4
    )

    # Model with specified backbone
    model = MultiTaskModel(
        backbone_name=backbone_name,
        pretrained=False,  # Loading from checkpoint
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)

    # Load checkpoint with backbone name
    checkpoint_path = os.path.join(
        config['train']['checkpoint_dir'],
        backbone_name,
        f"best_model_{backbone_name}.pth"
    )

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Evaluating with random weights.")

    model.eval()
    
    weather_preds, weather_targets = [], []
    time_preds, time_targets = [], []
    weather_logits_all, time_logits_all = [], []

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
            
            # Store logits for advanced metrics
            weather_logits_all.append(outputs['weather'].cpu())
            time_logits_all.append(outputs['timeofday'].cpu())
    
    # Concatenate all logits
    weather_logits_all = torch.cat(weather_logits_all, dim=0)
    time_logits_all = torch.cat(time_logits_all, dim=0)
    weather_targets_tensor = torch.tensor(weather_targets)
    time_targets_tensor = torch.tensor(time_targets)
    
    # Calculate advanced metrics
    weather_top3 = top_k_accuracy(weather_logits_all, weather_targets_tensor, k=3)
    time_top3 = top_k_accuracy(time_logits_all, time_targets_tensor, k=3)
    
    weather_conf = average_confidence(weather_logits_all, weather_targets_tensor)
    time_conf = average_confidence(time_logits_all, time_targets_tensor)
    
    weather_conf_by_class = confidence_by_class(weather_logits_all, weather_targets_tensor, num_weather_classes)
    time_conf_by_class = confidence_by_class(time_logits_all, time_targets_tensor, num_time_classes)

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
Backbone: {backbone_name}

Primary Metrics:
----------------
Weather Accuracy:       {weather_acc:.4f} ({weather_acc*100:.2f}%)
Time of Day Accuracy:   {time_acc:.4f} ({time_acc*100:.2f}%)

Advanced Metrics:
-----------------
Weather Top-3 Accuracy: {weather_top3:.4f} ({weather_top3*100:.2f}%)
Time Top-3 Accuracy:    {time_top3:.4f} ({time_top3*100:.2f}%)

Average Confidence (Correct Predictions):
------------------------------------------
Weather:                {weather_conf:.4f} ({weather_conf*100:.2f}%)
Time of Day:            {time_conf:.4f} ({time_conf*100:.2f}%)

Per-Class Confidence (Weather):
--------------------------------
"""
    
    for class_idx, class_name in enumerate(config['classes']['weather']):
        conf = weather_conf_by_class[class_idx]
        summary += f"  {class_name:<15}: {conf:.4f} ({conf*100:.2f}%)\n"
    
    summary += "\nPer-Class Confidence (Time of Day):\n"
    summary += "------------------------------------\n"
    
    for class_idx, class_name in enumerate(config['classes']['timeofday']):
        conf = time_conf_by_class[class_idx]
        summary += f"  {class_name:<15}: {conf:.4f} ({conf*100:.2f}%)\n"
    
    summary += "\n"

    print(summary)

    with open(os.path.join(report_dir, "summary.txt"), "w") as f:
        f.write(summary)
    
    # Additional prints as per instruction
    print(summary)
    print(weather_report)
    print(time_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate multi-task weather and time-of-day classifier')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='Backbone architecture to evaluate')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"Evaluating model with backbone: {args.backbone}")
    print("="*80)
    
    evaluate(backbone_name=args.backbone)
