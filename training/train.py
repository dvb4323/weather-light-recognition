import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import csv

from datasets.bdd_dataset import BDDDataset
from models.multitask_model import MultiTaskModel
from utils.metrics import compute_accuracy, AverageMeter

def train(backbone_name=None):
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # LOGIC ƯU TIÊN: CLI > Config YAML > Mặc định
    if backbone_name is None:
        if 'model' in config and 'backbone' in config['model']:
            backbone_name = config['model']['backbone']
        else:
            backbone_name = 'resnet18'
            print("Warning: 'backbone' not found in config.yaml. Using default 'resnet18'.")

    # Device setup
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"✅ Backbone selected: {backbone_name}")

    # --- TRANSFORMS ---
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=tuple(config['augmentation']['input_size']), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    # Datasets
    print("Initializing Datasets...")
    train_dataset = BDDDataset(
        img_dir=config['data']['train_images'],
        ann_dir=config['data']['train_anns'],
        transforms=train_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )

    val_dataset = BDDDataset(
        img_dir=config['data']['val_images'],
        ann_dir=config['data']['val_anns'],
        transforms=val_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    # Model
    model = MultiTaskModel(
        backbone_name=backbone_name,
        pretrained=config['model']['pretrained'],
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)

    print(f"Model created with {backbone_name} backbone")

    # --- MANUAL CLASS WEIGHTS ---
    w_weights = [0.4, 2.0, 1.5, 2.0, 2.0]
    weather_weights = torch.FloatTensor(w_weights).to(device)
    
    t_weights = [0.8, 3.0, 0.8]
    time_weights = torch.FloatTensor(t_weights).to(device)
    
    # Loss Function
    weather_criterion = nn.CrossEntropyLoss(weight=weather_weights)
    time_criterion = nn.CrossEntropyLoss(weight=time_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=1e-3)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # Training Setup
    best_val_loss = float('inf') # Chỉ giữ lại biến này
    
    num_epochs = config['train']['num_epochs']
    
    checkpoint_dir = os.path.join(config['train']['checkpoint_dir'], backbone_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # File Log
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "LR",
            "Train_Loss", "Train_Weather_Loss", "Train_Time_Loss", 
            "Train_Weather_Acc", "Train_Time_Acc", 
            "Val_Loss", "Val_Weather_Loss", "Val_Time_Loss",
            "Val_Weather_Acc", "Val_Time_Acc"
        ])

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = AverageMeter()
        train_loss_weather = AverageMeter()
        train_loss_time = AverageMeter()
        
        train_weather_acc = AverageMeter()
        train_time_acc = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, targets in pbar:
            images = images.to(device)
            weather_targets = targets['weather'].to(device)
            time_targets = targets['timeofday'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_w = weather_criterion(outputs['weather'], weather_targets)
            loss_t = time_criterion(outputs['timeofday'], time_targets)
            loss = loss_w + loss_t

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss.update(loss.item(), batch_size)
            train_loss_weather.update(loss_w.item(), batch_size)
            train_loss_time.update(loss_t.item(), batch_size)
            
            train_weather_acc.update(compute_accuracy(outputs['weather'], weather_targets), batch_size)
            train_time_acc.update(compute_accuracy(outputs['timeofday'], time_targets), batch_size)

            pbar.set_postfix({
                'loss': f"{train_loss.avg:.3f}",
                'loss_w': f"{train_loss_weather.avg:.3f}",
                'loss_t': f"{train_loss_time.avg:.3f}",
                'w_acc': f"{train_weather_acc.avg:.3f}",
                't_acc': f"{train_time_acc.avg:.3f}"
            })

        # Validation
        val_metrics = evaluate(model, val_loader, weather_criterion, time_criterion, device)
        (val_loss, val_loss_w, val_loss_t, val_w_acc, val_t_acc) = val_metrics
        
        # Cập nhật Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Ghi Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{current_lr:.1e}",
                f"{train_loss.avg:.4f}",
                f"{train_loss_weather.avg:.4f}",
                f"{train_loss_time.avg:.4f}",
                f"{train_weather_acc.avg:.4f}",
                f"{train_time_acc.avg:.4f}",
                f"{val_loss:.4f}",
                f"{val_loss_w:.4f}",
                f"{val_loss_t:.4f}",
                f"{val_w_acc:.4f}",
                f"{val_t_acc:.4f}"
            ])
            
        print(f"Val - Total Loss: {val_loss:.4f} | W_Acc: {val_w_acc:.4f} | T_Acc: {val_t_acc:.4f} | LR: {current_lr:.1e}")

        # --- [SỬA] CHỈ LƯU 1 BEST MODEL DỰA TRÊN VAL LOSS ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Đặt tên file là best_model_{backbone}.pth
            path_val = os.path.join(checkpoint_dir, f"best_model_{backbone_name}.pth")
            torch.save(model.state_dict(), path_val)
            print(f"🔥 Found new best model! Saved to: {path_val} (Val Loss: {best_val_loss:.4f})")

        # Lưu checkpoint định kỳ (giữ nguyên nếu muốn backup)
        if (epoch + 1) % config['train']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_{backbone_name}.pth"))

def evaluate(model, val_loader, weather_criterion, time_criterion, device):
    model.eval()
    losses = AverageMeter()
    losses_weather = AverageMeter()
    losses_time = AverageMeter()
    
    weather_accs = AverageMeter()
    time_accs = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            weather_targets = targets['weather'].to(device)
            time_targets = targets['timeofday'].to(device)

            outputs = model(images)

            loss_w = weather_criterion(outputs['weather'], weather_targets)
            loss_t = time_criterion(outputs['timeofday'], time_targets)
            loss = loss_w + loss_t

            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            losses_weather.update(loss_w.item(), batch_size)
            losses_time.update(loss_t.item(), batch_size)
            
            weather_accs.update(compute_accuracy(outputs['weather'], weather_targets), batch_size)
            time_accs.update(compute_accuracy(outputs['timeofday'], time_targets), batch_size)

    return losses.avg, losses_weather.avg, losses_time.avg, weather_accs.avg, time_accs.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multi-task weather and time-of-day classifier')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['resnet18', 'resnet34', 'resnet50', 
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='Backbone architecture to use')
    
    args = parser.parse_args()
    
    train(backbone_name=args.backbone)