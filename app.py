import streamlit as st
import numpy as np
import pickle
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("image_caption_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

@st.cache_resource
def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(base_model.input, base_model.get_layer('avg_pool').output)
    return model

def preprocess_image(img):
    img = img.resize((299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def extract_feature(image, model):
    img = preprocess_image(image)
    feature = model.predict(img, verbose=0)
    return feature

def generate_caption(model, tokenizer, photo, max_length=34):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('<start>', '').replace('<end>', '').strip()

# Streamlit UI
st.title("🖼️ Image Caption Generator")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        model, tokenizer = load_model_and_tokenizer()
        fe_model = load_feature_extractor()
        feature = extract_feature(image, fe_model)
        caption = generate_caption(model, tokenizer, feature)
    
    st.markdown("### 🧠 Generated Caption")
    st.success(caption)
