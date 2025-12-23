import torch
import yaml
from PIL import Image
from torchvision import transforms
import argparse
import json

from models.multitask_model import MultiTaskModel

def infer(image_path, model_path, config_path="config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

    # Transforms
    infer_transforms = transforms.Compose([
        transforms.Resize(tuple(config['augmentation']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['augmentation']['mean'], std=config['augmentation']['std'])
    ])

    # Model
    model = MultiTaskModel(
        num_weather_classes=config['model']['num_weather_classes'],
        num_time_classes=config['model']['num_time_classes']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process image
    image = Image.open(image_path).convert("RGB")
    input_tensor = infer_transforms(image).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        outputs = model(input_tensor)
        
    w_idx = outputs['weather'].argmax(dim=1).item()
    t_idx = outputs['timeofday'].argmax(dim=1).item()
    
    weather = config['classes']['weather'][w_idx]
    timeofday = config['classes']['timeofday'][t_idx]
    
    result = {
        "weather": weather,
        "timeofday": timeofday
    }
    
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    infer(args.image, args.model)
