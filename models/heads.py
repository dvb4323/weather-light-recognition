import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    A simple classification head consisting of a linear layer.
    """
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_heads(feature_dim, num_weather_classes, num_time_classes):
    """
    Returns the two classification heads.
    """
    weather_head = ClassificationHead(feature_dim, num_weather_classes)
    time_head = ClassificationHead(feature_dim, num_time_classes)
    return weather_head, time_head
