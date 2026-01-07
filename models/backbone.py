import torch.nn as nn
from torchvision import models

def get_backbone(model_name="resnet18", pretrained=True):
    """
    Get backbone model and feature dimension.
    
    Supported backbones:
    - resnet18, resnet34, resnet50
    - efficientnet_b0, efficientnet_b1, efficientnet_b2
    - mobilenet_v3_small, mobilenet_v3_large
    """
    
    # ResNet family
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        modules = list(model.children())[:-1]
        
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        feature_dim = model.fc.in_features
        modules = list(model.children())[:-1]
        
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        modules = list(model.children())[:-1]
    
    # EfficientNet family
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features
        modules = list(model.children())[:-1]
        
    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        feature_dim = model.classifier[1].in_features
        modules = list(model.children())[:-1]
        
    elif model_name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        feature_dim = model.classifier[1].in_features
        modules = list(model.children())[:-1]
    
    # MobileNet family
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        feature_dim = model.classifier[0].in_features
        modules = list(model.children())[:-1]
        
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        feature_dim = model.classifier[0].in_features
        modules = list(model.children())[:-1]
        
    else:
        raise ValueError(f"Unsupported backbone: {model_name}")
    
    backbone = nn.Sequential(*modules)
    return backbone, feature_dim
