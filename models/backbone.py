import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_backbone(model_name="resnet18", pretrained=True):
    if model_name == "resnet18":
        # Use new torchvision weights API
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove the fully connected layer
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        feature_dim = resnet.fc.in_features

    else:
        raise ValueError(f"Unsupported backbone: {model_name}")
        
    return backbone, feature_dim
