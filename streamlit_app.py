
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="StyleAI - Smart Fashion Finder",
    page_icon="👗",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8f0ff, #fff1f7);
}
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #7945d8;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
}
.product-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #eadcff;
    margin-bottom: 15px;
}
div.stButton > button {
    background: linear-gradient(90deg, #7945d8, #e64f9b);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

CATEGORIES = [
    "shirt", "t-shirt", "dress", "jeans",
    "trousers", "shoes", "sneakers",
    "handbag", "jacket", "skirt", "hat", "watch"
]

@st.cache_resource
def load_model():
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model

# ---------------- PRODUCTS ----------------

@st.cache_data
def load_products():
    file_path = Path(__file__).parent / "data" / "products.json"

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def find_products(category, products):
    category_map = {
        "t-shirt": "shirt",
        "trousers": "jeans",
        "sneakers": "shoes"
    }

    selected_category = category_map.get(category, category)

    return [
        product for product in products
        if product.get("category", "").lower() == selected_category
    ]

# ---------------- PREDICTION ----------------

def predict_image(image, processor, model):
    inputs = processor(
        text=[
            f"a photo of a {category}"
            for category in CATEGORIES
        ],
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    best_index = probabilities.argmax().item()

    category = CATEGORIES[best_index]
    confidence = probabilities[best_index].item() * 100

    return category, confidence

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">✨ StyleAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Fashion Identification & Product Recommendations</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("")

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("👗 StyleAI")
    st.write("Smart Fashion Assistant")
    st.divider()
    st.write("**Features**")
    st.write("📸 Upload a fashion image")
    st.write("🤖 AI clothing identification")
    st.write("🎯 Category-based recommendations")
    st.divider()
    st.caption("Powered by pretrained CLIP")

# ---------------- MAIN APP ----------------

st.header("Find Your Fashion Style")
st.write("Upload an image of clothing, shoes, or accessories.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("AI Identification")

        if st.button("🔍 Identify Fashion", use_container_width=True):

            try:
                with st.spinner("Loading AI model and analysing image..."):
                    processor, model = load_model()
                    category, confidence = predict_image(
                        image, processor, model
                    )

                st.success("Fashion identification completed!")

                st.metric("Identified Category", category.title())
                st.metric("Model Confidence", f"{confidence:.2f}%")

                products = load_products()
                recommendations = find_products(category, products)

                st.session_state["category"] = category
                st.session_state["confidence"] = confidence
                st.session_state["recommendations"] = recommendations

            except Exception as error:
                st.error(f"Something went wrong: {error}")

# ---------------- RECOMMENDATIONS ----------------

if "recommendations" in st.session_state:

    st.divider()
    st.header("🛍️ Recommended Products")

    recommendations = st.session_state["recommendations"]

    if recommendations:

        st.write(
            f"Products matching the category: "
            f"**{st.session_state['category'].title()}**"
        )

        columns = st.columns(3)

        for index, product in enumerate(recommendations):

            with columns[index % 3]:

                st.markdown('<div class="product-card">', unsafe_allow_html=True)

                st.subheader(
                    f"{product.get('emoji', '🛍️')} "
                    f"{product.get('name', 'Fashion Product')}"
                )

                st.write(product.get("description", ""))
                st.write(f"**Price: ₹{product.get('price', 'N/A')}**")

                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("No products found for this category yet.")

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center>Made with ❤️ using Python, Streamlit and CLIP</center>",
    unsafe_allow_html=True
)
