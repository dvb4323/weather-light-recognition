import torch.nn as nn
from torchvision import models

def get_backbone(model_name="resnet18", pretrained=True):
    """
    Returns a shared backbone model.
    """
    if model_name == "resnet18":
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the fully connected layer
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        feature_dim = resnet.fc.in_features
    else:
        raise ValueError(f"Unsupported backbone: {model_name}")
        
    return backbone, feature_dim
