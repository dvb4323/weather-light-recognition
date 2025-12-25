import torch
import yaml
from PIL import Image
from torchvision import transforms
import argparse
import json

from models.multitask_model import MultiTaskModel

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(config):
    return transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['mean'],
            std=config['augmentation']['std']
        )
    ])

def load_model(model_path, config, device):
    model = MultiTaskModel(
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference(model, input_tensor, config, device):
    """
    input_tensor: torch.Tensor of shape [1, 3, H, W]
    """
    with torch.no_grad():
        outputs = model(input_tensor)

    w_idx = outputs['weather'].argmax(dim=1).item()
    t_idx = outputs['timeofday'].argmax(dim=1).item()

    return {
        "weather": config['classes']['weather'][w_idx],
        "timeofday": config['classes']['timeofday'][t_idx]
    }

def infer(image_path, model_path, config_path="config.yaml"):
    config = load_config(config_path)

    device = torch.device(
        config['train']['device'] if torch.cuda.is_available() else "cpu"
    )

    transform = build_transforms(config)
    model = load_model(model_path, config, device)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    result = run_inference(model, input_tensor, config, device)

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    model_path = "./checkpoints/best_model.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=model_path, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    args = parser.parse_args()

    infer(args.image, args.model, args.config)
