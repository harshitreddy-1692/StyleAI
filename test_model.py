
import torch
from transformers import CLIPProcessor, CLIPModel

print("Loading CLIP AI model...")

model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

model.eval()

print("CLIP model loaded successfully!")
print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
print("AI model is ready for fashion image search!")
