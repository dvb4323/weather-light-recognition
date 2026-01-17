"""
Additional evaluation metrics for image classification.

Includes top-k accuracy and confidence analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np


def top_k_accuracy(outputs, targets, k=3):
    """
    Calculate top-k accuracy.
    
    Args:
        outputs: Model outputs (logits), shape (N, num_classes)
        targets: Ground truth labels, shape (N,)
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as float
    """
    with torch.no_grad():
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / targets.size(0)).item() / 100.0


def confidence_by_class(outputs, targets, num_classes):
    """
    Calculate average confidence for each class when correctly predicted.
    
    Args:
        outputs: Model outputs (logits), shape (N, num_classes)
        targets: Ground truth labels, shape (N,)
        num_classes: Number of classes
    
    Returns:
        Dict mapping class_idx -> average confidence
    """
    with torch.no_grad():
        probs = F.softmax(outputs, dim=1)
        pred_probs, preds = probs.max(dim=1)
        
        class_confidences = {}
        for class_idx in range(num_classes):
            # Get samples of this class that were correctly predicted
            class_mask = (targets == class_idx) & (preds == class_idx)
            if class_mask.sum() > 0:
                avg_conf = pred_probs[class_mask].mean().item()
                class_confidences[class_idx] = avg_conf
            else:
                class_confidences[class_idx] = 0.0
        
        return class_confidences


def average_confidence(outputs, targets):
    """
    Calculate overall average confidence for correct predictions.
    
    Args:
        outputs: Model outputs (logits), shape (N, num_classes)
        targets: Ground truth labels, shape (N,)
    
    Returns:
        Average confidence as float
    """
    with torch.no_grad():
        probs = F.softmax(outputs, dim=1)
        pred_probs, preds = probs.max(dim=1)
        
        correct_mask = (preds == targets)
        if correct_mask.sum() > 0:
            return pred_probs[correct_mask].mean().item()
        return 0.0
