import torch

from inference.infer import (
    load_model,
    load_config,
    run_inference
)


def test_inference():
    # Load config
    config = load_config("config.yaml")

    # Force CPU for CI
    device = torch.device("cpu")

    # Load model
    model = load_model("./checkpoints/best_model.pth", config, device)

    # Dummy input tensor (no transform, no PIL)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Run inference
    output = run_inference(model, dummy_input, config, device)

    # Assertions
    assert isinstance(output, dict)
    assert "weather" in output
    assert "timeofday" in output

    print("✅ Inference test passed")


if __name__ == "__main__":
    test_inference()
