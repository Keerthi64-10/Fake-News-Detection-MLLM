import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import os

# ----------------- CONFIG -----------------
MODEL_PATH = "best_fakeddit_clip.pth"   # change if in different folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP + your classifier
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

class MultimodalClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 2)
        )
    
    def forward(self, inputs):
        outputs = self.clip(**inputs)
        combined = torch.cat([outputs.text_embeds, outputs.image_embeds], dim=1)
        return self.classifier(combined)

model = MultimodalClassifier(clip_model).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print(f"✅ Model loaded on {DEVICE}")

# ----------------- PREDICTION FUNCTION -----------------
def predict_fake_news(title: str, image_input):
    """
    image_input can be:
      - PIL Image object
      - local file path (str)
      - image URL (str)
    """
    # Load image
    try:
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                response = requests.get(image_input, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}", 0.0, "Error"

    # Process
    inputs = processor(text=title, images=image, return_tensors="pt", 
                      padding="max_length", truncation=True, max_length=77)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs[0], dim=0)
        pred = outputs[0].argmax().item()
        confidence = probs[pred].item() * 100

    label = "FAKE" if pred == 1 else "REAL"
    return label, round(confidence, 2), probs

# ----------------- SIMPLE TEST -----------------
if __name__ == "__main__":
    print("\n=== Fakeddit Multimodal Detector ===\n")
    title = input("Enter news title: ")
    img_path = input("Enter image path or URL: ")
    
    label, conf, _ = predict_fake_news(title, img_path)
    print(f"\nPrediction: {label} ({conf}% confidence)")