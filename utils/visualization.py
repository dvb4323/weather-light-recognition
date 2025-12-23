import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def draw_predictions(image, predictions):
    """
    Overlays predictions on the image.
    image: PIL Image
    predictions: dict {'weather': '...', 'timeofday': '...'}
    """
    draw = ImageDraw.Draw(image)
    text = f"Weather: {predictions['weather']}\nTime: {predictions['timeofday']}"
    
    # Simple text placement
    draw.text((10, 10), text, fill="red")
    return image

def plot_prediction(image, predictions, title="Prediction"):
    plt.imshow(image)
    plt.title(f"W: {predictions['weather']} | T: {predictions['timeofday']}")
    plt.axis('off')
    plt.show()
