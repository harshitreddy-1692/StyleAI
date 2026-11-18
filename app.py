
from flask import Flask, render_template, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import json
import os

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load pretrained CLIP model
print("Loading StyleAI AI model...")

MODEL_NAME = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()

print("StyleAI AI model loaded successfully!")


# Fashion categories
categories = [
    "a photograph of a shirt",
    "a photograph of a t-shirt",
    "a photograph of a dress",
    "a photograph of jeans",
    "a photograph of trousers",
    "a photograph of shoes",
    "a photograph of sneakers",
    "a photograph of a handbag",
    "a photograph of a jacket",
    "a photograph of a skirt",
    "a photograph of a hat",
    "a photograph of a watch"
]


# Load product catalogue
def load_products():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "products.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Match identified category with catalogue categories
category_mapping = {
    "shirt": ["shirt", "t-shirt"],
    "t-shirt": ["shirt", "t-shirt"],
    "dress": ["dress"],
    "jeans": ["jeans"],
    "trousers": ["trousers"],
    "shoes": ["shoes", "sneakers"],
    "sneakers": ["shoes", "sneakers"],
    "handbag": ["handbag"],
    "jacket": ["jacket"],
    "skirt": ["skirt"],
    "hat": ["hat"],
    "watch": ["watch"]
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")

        inputs = processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=1)[0]

        best_index = probabilities.argmax().item()

        category = categories[best_index]
        confidence = float(probabilities[best_index]) * 100

        category = (
            category
            .replace("a photograph of a ", "")
            .replace("a photograph of an ", "")
        )

        category = category.strip().lower()

        # Find matching products
        products = load_products()

        matching_categories = category_mapping.get(
            category, [category]
        )

        similar_products = [
            product for product in products
            if product["category"].lower() in matching_categories
        ]

        return jsonify({
            "category": category.title(),
            "confidence": round(confidence, 2),
            "similar_products": similar_products,
            "product_count": len(similar_products),
            "message": "Fashion image identified successfully!"
        })

    except Exception as error:
        print("Prediction error:", error)

        return jsonify({
            "error": "Unable to process image. Please try again."
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
