import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import argparse
import torch.nn.functional as F

from datasets.bdd_dataset import BDDDataset
from models.multitask_model import MultiTaskModel

# --- 1. HÀM VẼ CONFUSION MATRIX ---
def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- 2. HÀM CHẠY INFERENCE ---
def get_predictions_with_confidence(model, loader, device):
    model.eval()
    
    w_preds, w_targets, w_probs = [], [], []
    t_preds, t_targets, t_probs = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Running Inference"):
            images = images.to(device)
            outputs = model(images)
            
            # Weather
            w_prob = F.softmax(outputs['weather'], dim=1)
            w_conf, w_p = torch.max(w_prob, dim=1)
            w_preds.extend(w_p.cpu().numpy())
            w_targets.extend(targets['weather'].numpy())
            w_probs.extend(w_conf.cpu().numpy())

            # Time
            t_prob = F.softmax(outputs['timeofday'], dim=1)
            t_conf, t_p = torch.max(t_prob, dim=1)
            t_preds.extend(t_p.cpu().numpy())
            t_targets.extend(targets['timeofday'].numpy())
            t_probs.extend(t_conf.cpu().numpy())
            
    return (
        np.array(w_targets), np.array(w_preds), np.array(w_probs),
        np.array(t_targets), np.array(t_preds), np.array(t_probs)
    )

def evaluate(backbone_name=None):
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Logic ưu tiên: CLI > Config > Mặc định
    if backbone_name is None:
        if 'model' in config and 'backbone' in config['model']:
            backbone_name = config['model']['backbone']
        else:
            backbone_name = 'resnet18'
            print("⚠️ Warning: 'backbone' not in config.yaml. Using default 'resnet18'.")

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"✅ Evaluating Backbone: {backbone_name}")

    # --- SETUP ĐƯỜNG DẪN ---
    # 1. Thư mục chứa Model
    ckpt_root_dir = os.path.join(config['train']['checkpoint_dir'], backbone_name)
    model_filename = f"best_model_{backbone_name}.pth"
    model_path = os.path.join(ckpt_root_dir, model_filename)

    # 2. Thư mục lưu kết quả đánh giá (Timestamp để không bị ghi đè)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_root_dir = os.path.join("evaluation", f"{backbone_name}_{timestamp}")
    report_dir = os.path.join(eval_root_dir, "reports")
    figure_dir = os.path.join(eval_root_dir, "figures")

    # Kiểm tra file model có tồn tại không
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at: {model_path}")
        print("Please run train.py first to generate the best model.")
        return

    # Tạo thư mục output
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print(f"📂 Loading Model from: {model_path}")
    print(f"📂 Saving Reports to: {eval_root_dir}")

    # --- DATA & MODEL SETUP ---
    val_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    # Dùng tập Test (hoặc Val nếu config trỏ vào Val)
    test_dataset = BDDDataset(
        img_dir=config['data']['test_images'], 
        ann_dir=config['data']['test_anns'],
        transforms=val_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)

    # Init Model
    model = MultiTaskModel(
        backbone_name=backbone_name, pretrained=False,
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)

    # Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # --- CHẠY INFERENCE ---
    w_true, w_pred, w_conf, t_true, t_pred, t_conf = get_predictions_with_confidence(model, test_loader, device)

    # --- TÍNH TOÁN METRICS ---
    # Weather Metrics
    w_acc = np.mean(w_true == w_pred)
    w_f1 = f1_score(w_true, w_pred, average='macro')
    
    # Time Metrics
    t_acc = np.mean(t_true == t_pred)
    t_f1 = f1_score(t_true, t_pred, average='macro')

    print(f"\n📊 Quick Result:")
    print(f"   - Weather: Acc={w_acc:.4f}, F1-Macro={w_f1:.4f}")
    print(f"   - Time:    Acc={t_acc:.4f}, F1-Macro={t_f1:.4f}")

    # --- TẠO BÁO CÁO CHI TIẾT ---
    print(f"\n📝 Generating detailed reports...")

    # 1. Text Reports (Classification Report)
    w_report_text = classification_report(w_true, w_pred, target_names=config['classes']['weather'])
    t_report_text = classification_report(t_true, t_pred, target_names=config['classes']['timeofday'])

    # 2. Confusion Matrices (Images)
    plot_confusion_matrix(
        confusion_matrix(w_true, w_pred), 
        config['classes']['weather'], 
        f"Weather Confusion Matrix\n({backbone_name})", 
        os.path.join(figure_dir, "weather_cm.png")
    )
    
    plot_confusion_matrix(
        confusion_matrix(t_true, t_pred), 
        config['classes']['timeofday'], 
        f"Time Confusion Matrix\n({backbone_name})", 
        os.path.join(figure_dir, "time_cm.png")
    )

    # 3. Summary Text File
    summary_content = f"""
Evaluation Report
=================
Date: {timestamp}
Model Backbone: {backbone_name}
Model Source: {model_path}

Global Metrics:
---------------
Weather Accuracy: {w_acc:.4f}
Weather Macro F1: {w_f1:.4f}
Avg Weather Confidence: {np.mean(w_conf):.4f}

Time Accuracy:    {t_acc:.4f}
Time Macro F1:    {t_f1:.4f}
Avg Time Confidence:    {np.mean(t_conf):.4f}

--------------------------------------------------
DETAILED WEATHER REPORT
--------------------------------------------------
{w_report_text}

--------------------------------------------------
DETAILED TIME REPORT
--------------------------------------------------
{t_report_text}
"""
    
    summary_path = os.path.join(report_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_content)

    print(f"✅ Done! Reports saved to: {eval_root_dir}")
    print(f"   - Summary: {summary_path}")
    print(f"   - Figures: {figure_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default=None)
    args = parser.parse_args()
    
    evaluate(backbone_name=args.backbone)