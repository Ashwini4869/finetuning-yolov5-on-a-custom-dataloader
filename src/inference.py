# Necessary imports
import torch
from PIL import Image

# Load the model
MODEL_PATH = "models/trained_model_final.pt"
model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")
model.names[0] = "face"  # Change the name of the labels to 'face'

# Load the image
IMAGE_PATH = "images/test.jpg"

image = Image.open(IMAGE_PATH)

# Run inference
results = model(image)

# Save the image
results.save(save_dir="images/results", labels=True, exist_ok=True)

print(results)
