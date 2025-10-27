
import streamlit as st
import numpy as np
import pickle
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------
# LOAD YOUR TRAINED MODEL
# ------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("image_caption_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

@st.cache_resource
def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model


def preprocess_image(img):
    img = img.resize((299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def extract_feature(image, model):
    img = preprocess_image(image)
    feature = model.predict(img, verbose=0)
    return feature


def generate_caption(model, tokenizer, photo, max_length=40):
    in_text = "<start>"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break
    caption = in_text.replace("<start>", "").replace("<end>", "").strip()
    return caption


# ------------------------------------------------------
# LOAD LIGHTWEIGHT PRETRAINED MODEL (ViT-GPT2)
# ------------------------------------------------------
@st.cache_resource
def load_lightweight_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer


def generate_lightweight_caption(image):
    model, feature_extractor, tokenizer = load_lightweight_model()
    image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=20, num_beams=3)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Caption Generator")

st.markdown(
    "Choose between your custom trained model or a lightweight pretrained model for generating captions."
)

# Model selection
model_choice = st.selectbox(
    "Select a Model",
    ["My Trained Model (InceptionV3 + LSTM)", "Lightweight Model (ViT-GPT2)"],
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Generating caption... please wait"):
        if model_choice == "My Trained Model (InceptionV3 + LSTM)":
            model, tokenizer = load_model_and_tokenizer()
            fe_model = load_feature_extractor()
            feature = extract_feature(image, fe_model)
            caption = generate_caption(model, tokenizer, feature)
        else:
            caption = generate_lightweight_caption(image)

    st.markdown("### ✨ Generated Caption:")
    st.success(caption)

    st.caption("⚙️ Powered by Streamlit + TensorFlow + Transformers")
