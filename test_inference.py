import torch
from PIL import Image
from inference.infer import load_model, run_inference

def test_inference():
    model = load_model("checkpoints/best_model.pth")
    img = Image.new("RGB", (224, 224))
    out = run_inference(model, img)
    assert "weather" in out
    assert "timeofday" in out
    print("Inference test passed")

if __name__ == "__main__":
    test_inference()
