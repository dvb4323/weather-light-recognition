import torch
import torch.nn as nn
from .backbone import get_backbone
from .heads import get_heads

class MultiTaskModel(nn.Module):
    """
    Multi-task model with a shared backbone and two classification heads.
    Handles different backbone architectures robustly.
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, 
                 num_weather_classes=6, num_time_classes=3):
        super(MultiTaskModel, self).__init__()
        
        self.backbone, self.feature_dim = get_backbone(backbone_name, pretrained)
        
        # Adaptive pooling to handle different output shapes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.weather_head = nn.Linear(self.feature_dim, num_weather_classes)
        self.time_head = nn.Linear(self.feature_dim, num_time_classes)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Handle different output shapes
        if len(features.shape) == 4:  # (B, C, H, W)
            features = self.adaptive_pool(features)
        
        features = torch.flatten(features, 1)
        
        # Pass through heads
        weather_logits = self.weather_head(features)
        time_logits = self.time_head(features)
        
        return {
            'weather': weather_logits,
            'timeofday': time_logits
        }
