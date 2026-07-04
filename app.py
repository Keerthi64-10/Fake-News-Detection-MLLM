import gradio as gr
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import re

# ----------------- CONFIG -----------------
MODEL_PATH = "best_fakeddit_crossattn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

class CrossAttentionFusion(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.text_to_image_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.1)
        self.image_to_text_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.1)
        
        self.fusion = nn.Sequential(
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, inputs):
        outputs = self.clip(**inputs)
        text_emb = outputs.text_embeds.unsqueeze(1)
        image_emb = outputs.image_embeds.unsqueeze(1)
        text_attended, _ = self.text_to_image_attn(text_emb, image_emb, image_emb)
        image_attended, _ = self.image_to_text_attn(image_emb, text_emb, text_emb)
        fused = torch.cat([text_emb.squeeze(1), image_emb.squeeze(1), 
                          text_attended.squeeze(1), image_attended.squeeze(1)], dim=1)
        return self.fusion(fused)

model = CrossAttentionFusion(clip_model).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("✅ Model loaded!")

# ====================== STRONG POLITICAL + FUTURE CLAIM DETECTION ======================
def strong_fact_check(title: str):
    title_lower = title.lower()
    current_year = 2026
    warnings = []
    severity = "normal"
    
    # Extract year
    year_match = re.search(r'20[2-9]\d', title)
    year = int(year_match.group(0)) if year_match else None
    
    # Political keywords
    political_keywords = ["chief minister", "cm", "minister", "election", "won", "wins", "became", "sworn", "government"]
    
    is_political = any(kw in title_lower for kw in political_keywords)
    
    # Rule 1: Future year + political claim = FAKE / SUSPICIOUS
    if year and year > current_year and is_political:
        warnings.append(f"⚠️ **HIGH RISK FAKE**: Claiming political event in {year} (future). This is very likely fabricated.")
        return warnings, "high"
    
    # Rule 2: Current year political claim (needs verification)
    if year == current_year and is_political and any(kw in title_lower for kw in ["became", "won", "sworn"]):
        warnings.append(f"⚠️ **NEEDS VERIFICATION**: {year} political claim. Cross-check with reliable news sources.")
        return warnings, "high"
    
    # Rule 3: Known actors becoming CM (Vijay, Rajinikanth, etc.)
    actor_to_cm = ["vijay", "rajinikanth", "kamal", "kamal haasan", "suriya"]
    for actor in actor_to_cm:
        if actor in title_lower and is_political:
            warnings.append(f"⚠️ **HIGH RISK**: Actor {actor.title()} becoming Chief Minister is a common fake news trope.")
            return warnings, "high"
    
    return warnings, severity

# ====================== PREDICTION ======================
def predict_fake_news(title: str, image_input):
    if not title.strip():
        return "**Please enter a title**", "", ""
    
    # Load image
    try:
        if isinstance(image_input, str) and image_input.startswith(("http", "https")):
            response = requests.get(image_input, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = image_input.convert("RGB") if isinstance(image_input, Image.Image) else Image.open(image_input).convert("RGB")
    except Exception as e:
        return f"❌ Image loading failed: {str(e)}", "", ""
    
    # Model prediction
    inputs = processor(text=title, images=image, return_tensors="pt", 
                      padding="max_length", truncation=True, max_length=77)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        raw_output = model(inputs)
        logits = raw_output.squeeze(0) if raw_output.dim() > 1 else raw_output
        temperature = 0.6   # Sharper confidence
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        pred = scaled_logits.argmax(dim=-1).item()
        confidence = probs[pred].item() * 100
    
    label = "FAKE" if pred == 0 else "REAL"
    
    # Strong Fact Check
    warnings, severity = strong_fact_check(title)
    
    if severity == "high":
        final_label = "**FAKE / SUSPICIOUS**"
        fact_msg = "\n".join(warnings)
    else:
        final_label = f"**{label}**"
        fact_msg = "✅ No strong red flags detected."
    
    model_conf = f"Model Confidence: **{confidence:.1f}%**"
    
    return final_label, model_conf, fact_msg

# ====================== GRADIO UI ======================
with gr.Blocks(title="Multimodal Fake News Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 Multimodal Fake News Detector\n**Text + Image + Strong Political Fact-Check**")
    
    with gr.Row():
        with gr.Column(scale=2):
            title_input = gr.Textbox(label="News Title / Headline", lines=3,
                                    placeholder="Vijay became the TN chief minister 2026")
            image_input = gr.Image(label="Upload Image", type="pil", height=320)
            url_input = gr.Textbox(label="Or paste Image URL")
        
        with gr.Column(scale=1):
            output_label = gr.Markdown(label="Final Verdict")
            output_conf = gr.Markdown(label="Model Confidence")
            output_fact = gr.Markdown(label="Fact-Check Analysis")
    
    submit_btn = gr.Button("🔍 Analyze", variant="primary", size="large")
    
    def combined_predict(title, image, url):
        if url and str(url).strip():
            return predict_fake_news(title, url)
        return predict_fake_news(title, image)
    
    submit_btn.click(
        fn=combined_predict,
        inputs=[title_input, image_input, url_input],
        outputs=[output_label, output_conf, output_fact]
    )

    gr.Markdown("**Note**: Political claims about future years or actors becoming CM are marked as **FAKE / SUSPICIOUS**.")

demo.launch(share=False)