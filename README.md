# 🖼️ Image Caption Generator

An intelligent image captioning application that automatically generates descriptive captions for images using deep learning models. The project offers two model options: a custom-trained InceptionV3 + LSTM model and a lightweight pretrained ViT-GPT2 model.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Sample Results](#sample-results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an image captioning system that combines Computer Vision and Natural Language Processing to automatically generate human-readable descriptions of images. The application provides a user-friendly web interface built with Streamlit, allowing users to upload images and receive captions instantly.

## ✨ Features

- **Dual Model Support**: Choose between custom-trained or pretrained models
- **Custom Model**: InceptionV3 (CNN) + LSTM architecture trained on custom dataset
- **Pretrained Model**: ViT-GPT2 model for quick and accurate captions
- **Interactive Web Interface**: Easy-to-use Streamlit application
- **Real-time Processing**: Generate captions in seconds
- **Image Upload**: Support for JPG, JPEG, and PNG formats
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🏗️ Model Architecture

### Custom Model (InceptionV3 + LSTM)
- **Feature Extractor**: InceptionV3 pretrained on ImageNet
  - Input: 299x299x3 RGB images
  - Output: 2048-dimensional feature vector
- **Caption Generator**: LSTM-based sequence model
  - Processes image features and generates word sequences
  - Uses custom tokenizer trained on caption dataset
  - Max caption length: 40 words

### Pretrained Model (ViT-GPT2)
- **Vision Encoder**: Vision Transformer (ViT)
- **Language Decoder**: GPT-2 architecture
- **Source**: `nlpconnect/vit-gpt2-image-captioning` from Hugging Face
- Optimized for speed and accuracy

## 📊 Performance

- **Average BLEU Score**: 0.1233 (across 10 samples)
- **Training Dataset**: Flickr8k (8,000 images)
- **Model Training**: Performed on Kaggle with GPU acceleration
- **System Limitations**: Due to computational constraints, training was limited to 8k images
- **Scalability**: With more powerful hardware (higher GPU memory, extended training time), the model can scale to:
  - Flickr30k (30,000 images) - Expected BLEU improvement to 0.25-0.30
  - MS COCO (100,000+ images) - Expected BLEU improvement to 0.35-0.45+
- **Inference Time**: 
  - Custom Model: ~2-3 seconds per image
  - Pretrained Model: ~1-2 seconds per image

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- (Optional) GPU for faster inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Files
Ensure you have the following files in your project directory:
- `image_caption_model.h5` - Your trained model weights
- `tokenizer.pkl` - Tokenizer for custom model

**For training your own model:**
```python
import kagglehub

# Download Flickr8k dataset
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)
```

*Note: The pretrained ViT-GPT2 model will be downloaded automatically on first use.*

## 💻 Usage

### Running the Application

1. **Start the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

3. **Generate Captions**:
   - Select your preferred model from the dropdown
   - Upload an image (JPG, JPEG, or PNG)
   - Wait for the caption to be generated
   - View your results!

### Example Usage

```python
# For programmatic usage
from PIL import Image
from app import generate_caption, load_model_and_tokenizer, extract_feature, load_feature_extractor

# Load models
model, tokenizer = load_model_and_tokenizer()
fe_model = load_feature_extractor()

# Load and process image
image = Image.open("sample.jpg")
feature = extract_feature(image, fe_model)

# Generate caption
caption = generate_caption(model, tokenizer, feature)
print(f"Caption: {caption}")
```

## 📁 Project Structure

```
image-caption-generator/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── image_caption_model.h5      # Trained model weights
├── tokenizer.pkl              # Tokenizer for custom model
├── training_notebook.ipynb    # Model training notebook (Kaggle)
├── README.md                  # Project documentation
│
## 🎓 Model Training

The custom model was trained using the following approach:

### Dataset
- **Dataset**: Flickr8k - 8,000 images with 5 captions each
- **Source**: Available on Kaggle
- **Access Method**:
```python
import kagglehub
# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)
```
- **Dataset Link**: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Images preprocessed to 299x299 resolution
- Captions tokenized with special start/end tokens
- **Note**: Due to system limitations, training was performed on the full 8k images. With more powerful hardware (higher GPU memory, longer training time), the model can be trained on larger datasets like Flickr30k (30,000 images) or MS COCO (>100,000 images) for significantly improved accuracy and BLEU scores.

### Training Configuration
- **Framework**: TensorFlow/Keras
- **Feature Extractor**: InceptionV3 (frozen weights)
- **Encoder-Decoder Architecture**: LSTM-based
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Training Platform**: Kaggle GPU

### Training Process
Refer to `training_notebook.ipynb` for complete training code and detailed steps.

## 🖼️ Sample Results

### Example 1
**Image**: <img width="966" height="1237" alt="model_output5" src="https://github.com/user-attachments/assets/69e3906f-f5c1-4d79-bcd1-29cd855b52da" />
 
**Generated Caption**: "a snowmobiler flies through the air end"

### Example 2
**Image**: <img width="850" height="1219" alt="model_output2" src="https://github.com/user-attachments/assets/9cfe090a-d6eb-49c6-8e41-7c996321aad9" />

**Generated Caption**: "three dogs are playing in a grassy field end"

### Example 3
**Image**: <img width="710" height="1270" alt="output3" src="https://github.com/user-attachments/assets/b18d17c8-ef2b-4485-b029-cda4bdd13bb6" />

**Generated Caption**: "a man in a red jacket is climbing a snowy mountain end"

*Note: Place sample images in the `sample_images/` folder for testing*

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Deep Learning**: 
  - TensorFlow 2.17.0
  - PyTorch
  - Transformers (Hugging Face)
- **Computer Vision**: 
  - InceptionV3
  - Vision Transformer (ViT)
- **NLP**: 
  - LSTM
  - GPT-2
- **Image Processing**: Pillow (PIL)
- **Numerical Computing**: NumPy

## 🔮 Future Improvements

- [ ] Train on larger datasets (Flickr30k, MS COCO) with better hardware
- [ ] Improve BLEU score through extended training epochs
- [ ] Add attention mechanism visualization
- [ ] Support for batch image processing
- [ ] Deploy on cloud platforms (AWS, Azure, Heroku)
- [ ] Add multilingual caption support
- [ ] Implement beam search for better captions
- [ ] Add confidence scores for generated captions
- [ ] Create REST API endpoint
- [ ] Add caption editing and feedback mechanism
- [ ] Fine-tune hyperparameters for optimal performance
- [ ] Implement transfer learning with newer architectures (EfficientNet, ResNet)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flickr8k dataset from Kaggle: [adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- InceptionV3 model from TensorFlow/Keras
- ViT-GPT2 model from Hugging Face (nlpconnect)
- Kaggle for providing free GPU training infrastructure
- Streamlit for the amazing web framework

**Note**: This project demonstrates image captioning capabilities within system constraints. With access to more powerful hardware (e.g., Tesla V100, A100 GPUs) and larger datasets, the model performance can be significantly enhanced.

## 📧 Contact

Divy Dobariya

Email: divydobariya11@gmail.com
LinkedIn: linkedin.com/in/divy-dobariya-92881423b
GitHub: @Divy005

Project Link: https://github.com/Divy005/image_caption_generator

---

⭐ If you found this project helpful, please give it a star!

**Made with ❤️ and Python**
