import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from datasets.bdd_dataset import BDDDataset
from models.multitask_model import MultiTaskModel
from utils.metrics import compute_accuracy, AverageMeter

def train(backbone_name='resnet18'):
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Device setup
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Backbone: {backbone_name}")

    # Transforms - Original simple approach that worked
    train_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    # Datasets & DataLoaders
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

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4)

    # Model with specified backbone
    model = MultiTaskModel(
        backbone_name=backbone_name,
        pretrained=config['model']['pretrained'],
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)

    print(f"Model created with {backbone_name} backbone")

    # Loss & Optimizer - Original simple approach that worked
    # All classes weight 1.0, except undefined gets low weight
    num_weather_classes = config['model']['num_weather_classes']
    num_time_classes = config['model']['num_time_classes']
    
    # Weather weights: [1.0, 1.0, 1.0, 1.0, 1.0, 0.01]
    weather_weights = torch.ones(num_weather_classes)
    weather_weights[-1] = 0.01  # undefined
    
    # Time weights: [1.0, 1.0, 1.0, 0.01]
    time_weights = torch.ones(num_time_classes)
    time_weights[-1] = 0.01  # undefined
    
    print(f"Weather class weights: {weather_weights.tolist()}")
    print(f"Time class weights: {time_weights.tolist()}")
    
    weather_criterion = nn.CrossEntropyLoss(weight=weather_weights.to(device))
    time_criterion = nn.CrossEntropyLoss(weight=time_weights.to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

    # Training Loop
    best_val_loss = float('inf')
    num_epochs = config['train']['num_epochs']
    
    # Create checkpoint directory with backbone name
    checkpoint_dir = os.path.join(config['train']['checkpoint_dir'], backbone_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = AverageMeter()
        train_weather_acc = AverageMeter()
        train_time_acc = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, targets in pbar:
            images = images.to(device)
            weather_targets = targets['weather'].to(device)
            time_targets = targets['timeofday'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_weather = weather_criterion(outputs['weather'], weather_targets)
            loss_time = time_criterion(outputs['timeofday'], time_targets)
            loss = loss_weather + loss_time

            loss.backward()
            optimizer.step()

            # Metrics
            train_loss.update(loss.item(), images.size(0))
            train_weather_acc.update(compute_accuracy(outputs['weather'], weather_targets), images.size(0))
            train_time_acc.update(compute_accuracy(outputs['timeofday'], time_targets), images.size(0))

            pbar.set_postfix({'loss': f"{train_loss.avg:.4f}", 'w_acc': f"{train_weather_acc.avg:.4f}", 't_acc': f"{train_time_acc.avg:.4f}"})

        # Validation
        val_loss, val_w_acc, val_t_acc = evaluate(model, val_loader, weather_criterion, time_criterion, device)
        print(f"Val - Loss: {val_loss:.4f}, Weather Acc: {val_w_acc:.4f}, Time Acc: {val_t_acc:.4f}")

        # Checkpoint with backbone name
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_{backbone_name}.pth"))
            print(f"Saved best model: best_model_{backbone_name}.pth")

        if (epoch + 1) % config['train']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_{backbone_name}.pth"))


def evaluate(model, val_loader, weather_criterion, time_criterion, device):
    model.eval()
    losses = AverageMeter()
    weather_accs = AverageMeter()
    time_accs = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            weather_targets = targets['weather'].to(device)
            time_targets = targets['timeofday'].to(device)

            outputs = model(images)

            loss_weather = weather_criterion(outputs['weather'], weather_targets)
            loss_time = time_criterion(outputs['timeofday'], time_targets)
            loss = loss_weather + loss_time

            losses.update(loss.item(), images.size(0))
            weather_accs.update(compute_accuracy(outputs['weather'], weather_targets), images.size(0))
            time_accs.update(compute_accuracy(outputs['timeofday'], time_targets), images.size(0))

    return losses.avg, weather_accs.avg, time_accs.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multi-task weather and time-of-day classifier')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='Backbone architecture to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"Training with backbone: {args.backbone}")
    print("="*80)
    
    train(backbone_name=args.backbone)
