# src/inference.py

import os
import json
import clip
import torch
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = "data/raw"
prompt_file = "data/prompts.txt"
output_csv = "outputs/results/batch_predictions.csv"
json_dir = "outputs/results/json"
output_fig_dir = "outputs/figures"

os.makedirs(output_fig_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# ─── INITIALIZE WANDB ──────────────────────────────────────────────────────────

wandb.init(
    project="zero-shot-urban-scene-classification",
    name="batch-inference",
    config={
        "model": "ViT-B/32",
        "device": device
    }
)

# ─── LOAD MODEL & PROMPTS ──────────────────────────────────────────────────────

model, preprocess = clip.load("ViT-B/32", device=device)

with open(prompt_file, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]
text_tokens = clip.tokenize(prompts).to(device)

# ─── INFERENCE LOOP ────────────────────────────────────────────────────────────

results = []

for fname in tqdm(os.listdir(image_dir)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, fname)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    top_idx = probs.argmax()
    top_prediction = prompts[top_idx]
    confidence = float(probs[top_idx])

    # Log to results
    result_row = {
        "image": fname,
        "top_prediction": top_prediction,
        "confidence": round(confidence, 4),
        **{f"prob_{prompt}": round(float(p), 4) for prompt, p in zip(prompts, probs)}
    }
    results.append(result_row)

    # Save prediction figure
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"{top_prediction} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_fig_dir, f"{fname.rsplit('.', 1)[0]}_pred.png")
    plt.savefig(save_path)
    plt.close()

    # Save JSON
    json_path = os.path.join(json_dir, f"{fname.rsplit('.', 1)[0]}_result.json")
    with open(json_path, "w") as jf:
        json.dump({
            "image": fname,
            "top_prediction": top_prediction,
            "confidence": confidence,
            "all_probabilities": {p: float(prob) for p, prob in zip(prompts, probs)}
        }, jf, indent=2)

    # Log to wandb
    wandb.log({
        "image_name": fname,
        "top_prediction": top_prediction,
        "confidence": confidence,
        "image": wandb.Image(img_path, caption=f"{top_prediction} ({confidence*100:.1f}%)"),
        "probabilities": {
            prompt: float(prob) for prompt, prob in zip(prompts, probs)
        }
    })

# ─── SAVE FINAL OUTPUTS ────────────────────────────────────────────────────────

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

artifact = wandb.Artifact("inference_outputs", type="inference")
artifact.add_file(output_csv)
wandb.log_artifact(artifact)

wandb.finish()

print(f"\n Inference complete. Results saved to: {output_csv}")
