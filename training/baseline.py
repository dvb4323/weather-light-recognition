"""
Baseline Comparison Script

Calculates simple baselines for weather and time-of-day classification:
1. Random Guessing
2. Majority Class
3. Stratified Random (weighted by class distribution)

Usage: python -m training.baseline
"""

import yaml
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from datasets.bdd_dataset import BDDDataset


def calculate_class_distribution(dataset, task='weather'):
    """Calculate class distribution in dataset."""
    labels = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        labels.append(targets[task].item())
    
    counter = Counter(labels)
    total = len(labels)
    
    distribution = {}
    for class_idx in sorted(counter.keys()):
        count = counter[class_idx]
        percentage = count / total * 100
        distribution[class_idx] = {
            'count': count,
            'percentage': percentage
        }
    
    return distribution, labels


def majority_class_baseline(labels, majority_class):
    """Calculate accuracy if always predicting majority class."""
    predictions = [majority_class] * len(labels)
    correct = sum(1 for pred, true in zip(predictions, labels) if pred == true)
    accuracy = correct / len(labels)
    return accuracy


def random_baseline(labels, num_classes):
    """Calculate expected accuracy for random guessing."""
    return 1.0 / num_classes


def stratified_random_baseline(labels, distribution):
    """Calculate expected accuracy for stratified random guessing."""
    # Expected accuracy = sum of (p_i)^2 for each class
    expected_acc = sum((dist['percentage'] / 100) ** 2 for dist in distribution.values())
    return expected_acc


def evaluate_baselines(config_path='config.yaml'):
    """Evaluate all baselines on test set."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    
    # Simple transforms (no augmentation for test)
    test_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], 
                           std=config['augmentation']['std'])
    ])
    
    # Load test dataset
    test_dataset = BDDDataset(
        img_dir=config['data']['test_images'],
        ann_dir=config['data']['test_anns'],
        transforms=test_transforms,
        weather_classes=config['classes']['weather'],
        time_classes=config['classes']['timeofday']
    )
    
    print(f"\nTest set size: {len(test_dataset)} images")
    print("="*80)
    
    # ========== WEATHER BASELINES ==========
    print("\n📊 WEATHER CLASSIFICATION BASELINES")
    print("-"*80)
    
    weather_dist, weather_labels = calculate_class_distribution(test_dataset, 'weather')
    num_weather_classes = len(config['classes']['weather'])
    
    print("\nClass Distribution:")
    for class_idx, class_name in enumerate(config['classes']['weather']):
        if class_idx in weather_dist:
            dist = weather_dist[class_idx]
            print(f"  {class_name:<15}: {dist['count']:>5} samples ({dist['percentage']:>5.2f}%)")
    
    # Find majority class
    majority_weather_class = max(weather_dist.items(), key=lambda x: x[1]['count'])[0]
    majority_weather_name = config['classes']['weather'][majority_weather_class]
    
    print(f"\nMajority class: {majority_weather_name} (class {majority_weather_class})")
    
    # Calculate baselines
    random_acc = random_baseline(weather_labels, num_weather_classes)
    majority_acc = majority_class_baseline(weather_labels, majority_weather_class)
    stratified_acc = stratified_random_baseline(weather_labels, weather_dist)
    
    print("\n" + "="*80)
    print("WEATHER BASELINE RESULTS:")
    print("="*80)
    print(f"1. Random Guessing:        {random_acc:.4f} ({random_acc*100:.2f}%)")
    print(f"2. Stratified Random:      {stratified_acc:.4f} ({stratified_acc*100:.2f}%)")
    print(f"3. Majority Class:         {majority_acc:.4f} ({majority_acc*100:.2f}%)")
    print("="*80)
    
    # ========== TIME-OF-DAY BASELINES ==========
    print("\n📊 TIME-OF-DAY CLASSIFICATION BASELINES")
    print("-"*80)
    
    time_dist, time_labels = calculate_class_distribution(test_dataset, 'timeofday')
    num_time_classes = len(config['classes']['timeofday'])
    
    print("\nClass Distribution:")
    for class_idx, class_name in enumerate(config['classes']['timeofday']):
        if class_idx in time_dist:
            dist = time_dist[class_idx]
            print(f"  {class_name:<15}: {dist['count']:>5} samples ({dist['percentage']:>5.2f}%)")
    
    # Find majority class
    majority_time_class = max(time_dist.items(), key=lambda x: x[1]['count'])[0]
    majority_time_name = config['classes']['timeofday'][majority_time_class]
    
    print(f"\nMajority class: {majority_time_name} (class {majority_time_class})")
    
    # Calculate baselines
    random_acc_time = random_baseline(time_labels, num_time_classes)
    majority_acc_time = majority_class_baseline(time_labels, majority_time_class)
    stratified_acc_time = stratified_random_baseline(time_labels, time_dist)
    
    print("\n" + "="*80)
    print("TIME-OF-DAY BASELINE RESULTS:")
    print("="*80)
    print(f"1. Random Guessing:        {random_acc_time:.4f} ({random_acc_time*100:.2f}%)")
    print(f"2. Stratified Random:      {stratified_acc_time:.4f} ({stratified_acc_time*100:.2f}%)")
    print(f"3. Majority Class:         {majority_acc_time:.4f} ({majority_acc_time*100:.2f}%)")
    print("="*80)
    
    # ========== COMPARISON WITH YOUR MODELS ==========
    print("\n" + "="*80)
    print("COMPARISON WITH YOUR MODELS")
    print("="*80)
    
    # You can update these with your actual results
    resnet18_weather = 0.7517
    resnet18_time = 0.9365
    efficientnet_weather = 0.7608
    efficientnet_time = 0.9380
    
    print("\n📊 WEATHER CLASSIFICATION:")
    print("-"*80)
    print(f"{'Method':<30} {'Accuracy':<15} {'Improvement vs Majority'}")
    print("-"*80)
    print(f"{'Random Guessing':<30} {random_acc*100:>6.2f}%        -")
    print(f"{'Stratified Random':<30} {stratified_acc*100:>6.2f}%        -")
    print(f"{'Majority Class (Baseline)':<30} {majority_acc*100:>6.2f}%        -")
    print(f"{'ResNet18':<30} {resnet18_weather*100:>6.2f}%        +{(resnet18_weather - majority_acc)/majority_acc*100:.1f}%")
    print(f"{'EfficientNet-B0':<30} {efficientnet_weather*100:>6.2f}%        +{(efficientnet_weather - majority_acc)/majority_acc*100:.1f}%")
    
    print("\n📊 TIME-OF-DAY CLASSIFICATION:")
    print("-"*80)
    print(f"{'Method':<30} {'Accuracy':<15} {'Improvement vs Majority'}")
    print("-"*80)
    print(f"{'Random Guessing':<30} {random_acc_time*100:>6.2f}%        -")
    print(f"{'Stratified Random':<30} {stratified_acc_time*100:>6.2f}%        -")
    print(f"{'Majority Class (Baseline)':<30} {majority_acc_time*100:>6.2f}%        -")
    print(f"{'ResNet18':<30} {resnet18_time*100:>6.2f}%        +{(resnet18_time - majority_acc_time)/majority_acc_time*100:.1f}%")
    print(f"{'EfficientNet-B0':<30} {efficientnet_time*100:>6.2f}%        +{(efficientnet_time - majority_acc_time)/majority_acc_time*100:.1f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Weather: Your best model (EfficientNet-B0) achieves {efficientnet_weather*100:.2f}%")
    print(f"   → {(efficientnet_weather - majority_acc)/majority_acc*100:.1f}% improvement over majority class baseline")
    print(f"   → {(efficientnet_weather - random_acc)/random_acc*100:.1f}% improvement over random guessing")
    
    print(f"\n✅ Time: Your best model (EfficientNet-B0) achieves {efficientnet_time*100:.2f}%")
    print(f"   → {(efficientnet_time - majority_acc_time)/majority_acc_time*100:.1f}% improvement over majority class baseline")
    print(f"   → {(efficientnet_time - random_acc_time)/random_acc_time*100:.1f}% improvement over random guessing")
    
    print("\n" + "="*80)
    
    return {
        'weather': {
            'random': random_acc,
            'stratified': stratified_acc,
            'majority': majority_acc,
            'resnet18': resnet18_weather,
            'efficientnet_b0': efficientnet_weather
        },
        'time': {
            'random': random_acc_time,
            'stratified': stratified_acc_time,
            'majority': majority_acc_time,
            'resnet18': resnet18_time,
            'efficientnet_b0': efficientnet_time
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate baseline accuracies')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    results = evaluate_baselines(args.config)
