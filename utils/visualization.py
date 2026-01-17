"""
Visualization utilities for training monitoring.

Generates separate training plots and saves training history.
"""

import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


def plot_training_curves(history, save_dir, backbone_name):
    """
    Plot training curves as separate files: losses and accuracies.
    
    Args:
        history: Dict with training history
        save_dir: Directory to save plots
        backbone_name: Name of backbone for title
    
    Returns:
        List of saved plot paths
    """
    epochs = [e['epoch'] for e in history['epochs']]
    
    # Create timestamped folder for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(save_dir, f'training_plots_{backbone_name}_{timestamp}')
    os.makedirs(plot_dir, exist_ok=True)
    
    saved_plots = []
    
    # 1. Total Loss (Combined)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [e['train_loss'] for e in history['epochs']], 
             'b-', label='Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, [e['val_loss'] for e in history['epochs']], 
             'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Total Loss (Weather + Time) - {backbone_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, '1_total_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_path)
    
    # 2. Weather Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [e['train_weather_loss'] for e in history['epochs']], 
             'b-', label='Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, [e['val_weather_loss'] for e in history['epochs']], 
             'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Weather Classification Loss - {backbone_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, '2_weather_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_path)
    
    # 3. Time Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [e['train_time_loss'] for e in history['epochs']], 
             'b-', label='Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, [e['val_time_loss'] for e in history['epochs']], 
             'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Time-of-Day Classification Loss - {backbone_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, '3_time_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_path)
    
    # 4. Weather Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [e['train_weather_acc'] for e in history['epochs']], 
             'b-', label='Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, [e['val_weather_acc'] for e in history['epochs']], 
             'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Weather Classification Accuracy - {backbone_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, '4_weather_accuracy.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_path)
    
    # 5. Time Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [e['train_time_acc'] for e in history['epochs']], 
             'b-', label='Train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, [e['val_time_acc'] for e in history['epochs']], 
             'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Time-of-Day Classification Accuracy - {backbone_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, '5_time_accuracy.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_path)
    
    print(f"✓ Saved {len(saved_plots)} training plots to: {plot_dir}")
    return saved_plots


def save_training_history(history, save_dir, backbone_name):
    """
    Save training history to JSON file in timestamped folder.
    
    Args:
        history: Dict with training history
        save_dir: Directory to save history
        backbone_name: Name of backbone
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join(save_dir, f'training_plots_{backbone_name}_{timestamp}')
    os.makedirs(history_dir, exist_ok=True)
    
    history_path = os.path.join(history_dir, 'training_history.json')
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Saved training history to: {history_path}")
    return history_path


def print_training_summary(history):
    """
    Print a summary of training results.
    
    Args:
        history: Dict with training history
    """
    best_epoch_idx = history['best_epoch'] - 1
    best_epoch_data = history['epochs'][best_epoch_idx]
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Backbone: {history['backbone']}")
    print(f"Total Epochs: {len(history['epochs'])}")
    print(f"Best Epoch: {history['best_epoch']}")
    print(f"\nBest Model Performance (Epoch {history['best_epoch']}):")
    print(f"  Total Loss:             {best_epoch_data['val_loss']:.4f}")
    print(f"  Weather Loss:           {best_epoch_data['val_weather_loss']:.4f}")
    print(f"  Time Loss:              {best_epoch_data['val_time_loss']:.4f}")
    print(f"  Weather Accuracy:       {best_epoch_data['val_weather_acc']:.4f} ({best_epoch_data['val_weather_acc']*100:.2f}%)")
    print(f"  Time-of-Day Accuracy:   {best_epoch_data['val_time_acc']:.4f} ({best_epoch_data['val_time_acc']*100:.2f}%)")
    print("="*80 + "\n")

