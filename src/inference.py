# src/inference.py

import clip
import torch
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load image
image_path = "data/raw/Picture1.png"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Define class prompts
prompts = [
    "a photo of a busy intersection",
    "a photo of a stop sign",
    "a photo of a pedestrian crossing",
    "a photo of a highway",
    "a photo of a green traffic light"
]

# Tokenize prompts
text = clip.tokenize(prompts).to(device)

# Run inference
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Print result
for label, prob in zip(prompts, probs[0]):
    print(f"{label}: {prob:.4f}")

# Display image
plt.imshow(Image.open(image_path))
plt.title(f"Prediction: {prompts[probs[0].argmax()]}")
plt.axis('off')
plt.savefig("outputs/results/Picture_prediction.png", bbox_inches='tight')

# Save the image with prediction
df = pd.DataFrame({
    "Prompt": prompts,
    "Probability": probs[0]
})
df.to_csv("outputs/results/Picture_prediction.csv", index=False)

output = {
    "file": image_path,
    "top_prediction": prompts[probs[0].argmax()],
    "probabilities": {label: float(prob) for label, prob in zip(prompts, probs[0])}
}

with open("outputs/results/Picture_result.json", "w") as f:
    json.dump(output, f, indent=2)

