# src/inference.py

import os
import clip
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


# Configs
device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = "data/raw"
prompt_file = "data/prompts.txt"
output_csv = "outputs/results/batch_predictions.csv"
output_dir = "outputs/figures"

# Load model and preprocess
model, preprocess = clip.load("ViT-B/32", device=device)

# Load prompts from file
with open(prompt_file, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]
text_tokens = clip.tokenize(prompts).to(device)

# Prepare results
results = []

# Process all images in folder
os.makedirs(output_dir, exist_ok=True)
for fname in tqdm(os.listdir(image_dir)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, fname)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    top_idx = probs.argmax()
    results.append({
        "image": fname,
        "top_prediction": prompts[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        **{f"prob_{prompt}": round(float(p), 4) for prompt, p in zip(prompts, probs)}
    })

for row in results:
    img_file = os.path.join(image_dir, row["image"])
    img = Image.open(img_file)

    plt.imshow(img)
    plt.title(f"{row['top_prediction']} ({row['confidence']*100:.1f}%)")
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{row['image'].rsplit('.', 1)[0]}_pred.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction image: {save_path}")

# Directory for JSON results
json_dir = "outputs/results/json"
os.makedirs(json_dir, exist_ok=True)

# Save one JSON file per image
for row in results:
    json_path = os.path.join(json_dir, f"{row['image'].rsplit('.', 1)[0]}_result.json")
    json_data = {
        "image": row["image"],
        "top_prediction": row["top_prediction"],
        "confidence": row["confidence"],
        "all_probabilities": {
            k.replace("prob_", ""): v for k, v in row.items() if k.startswith("prob_")
        }
    }
    with open(json_path, "w") as jf:
        json.dump(json_data, jf, indent=2)

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"\n Inference complete. Results saved to: {output_csv}")
