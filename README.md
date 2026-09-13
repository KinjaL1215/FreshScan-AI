# 🍎 FreshScan AI — Food Quality Analyzer

**FreshScan AI** is a deep-learning-based web application that analyzes images of fruits and vegetables and predicts whether the detected food appears **Fresh** or **Rotten**.

The application uses a **MobileNetV2 transfer-learning model** trained on fresh and rotten fruit/vegetable images and provides the predicted food, freshness probabilities, and AI confidence.

## 🚀 Live Demo

👉 **[Open FreshScan AI](https://freshscan-ai.streamlit.app/)**

## ✨ Features

* 📸 Take a food photo directly from the camera
* 📁 Upload JPG, JPEG, or PNG images
* 🤖 Deep-learning based food classification
* 🥗 Fresh vs Rotten quality prediction
* 📊 Fresh and Rotten probability percentages
* 🎯 AI prediction confidence
* 🌈 Interactive Streamlit interface
* ⚡ Fast image prediction using MobileNetV2
* 🍎 Supports multiple fruits and vegetables

## 🧠 Machine Learning Model

The project uses **MobileNetV2** with transfer learning from ImageNet.

### Model Architecture

```text
Input Image
    ↓
Resize to 224 × 224
    ↓
MobileNetV2
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Layer
    ↓
Softmax
    ↓
26 Food Classes
```

### Model Details

* **Architecture:** MobileNetV2
* **Input Size:** 224 × 224 × 3
* **Number of Classes:** 26
* **Optimizer:** Adam
* **Loss:** Sparse Categorical Crossentropy
* **Data Augmentation:** Flip, Rotation, Zoom, Contrast
* **Transfer Learning:** ImageNet
* **Best Validation Accuracy:** 92.53%
* **Test Accuracy:** 91.98%

## 🥗 Supported Foods

### Fruits

* 🍎 Apple
* 🍌 Banana
* 🥭 Mango
* 🍊 Orange
* 🍓 Strawberry

### Vegetables

* 🫑 Bellpepper
* 🫑 Capsicum
* 🥒 Bittergourd
* 🥕 Carrot
* 🥒 Cucumber
* 🌱 Okra
* 🥔 Potato
* 🍅 Tomato

Each food has two quality classes:

```text
Fresh
Rotten
```

## 📊 Model Performance

The trained model was evaluated on **7,132 unseen test images**.

| Metric            |  Score |
| ----------------- | -----: |
| Test Accuracy     | 91.98% |
| Macro Precision   | 89.80% |
| Macro Recall      | 92.58% |
| Macro F1-Score    | 90.91% |
| Weighted F1-Score | 92.13% |

## 🛠️ Tech Stack

### Machine Learning

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Scikit-learn

### Web Application

* Streamlit
* Python
* Pillow

### Deployment

* GitHub
* Streamlit Community Cloud

## 📂 Project Structure

```text
freshscan-ai/
│
├── models/
│   └── best_model.keras
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/freshscan-ai.git
cd freshscan-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

```text
streamlit
tensorflow==2.21.0
numpy
pillow
```

## 🔬 How It Works

1. User uploads a food image or takes a photo.
2. The image is converted to RGB.
3. The image is resized to `224 × 224`.
4. The image is normalized.
5. MobileNetV2 processes the image.
6. The model predicts one of the 26 food-quality classes.
7. The application identifies:

   * Food type
   * Fresh/Rotten condition
   * Fresh probability
   * Rotten probability
   * Overall model confidence
8. The result is displayed through the Streamlit interface.

## ⚠️ Disclaimer

FreshScan AI is an **image-based machine-learning classification system**.

The displayed Fresh/Rotten percentages represent the model's visual prediction probabilities. They are **not laboratory measurements** and should not be considered a definitive food-safety or contamination test.

## 🎯 Future Improvements

* 🔥 Grad-CAM visual explanations
* 📱 Improved mobile interface
* 🧠 Fine-tuning MobileNetV2
* 📈 More balanced training data
* 🍇 Support for additional fruits and vegetables
* 🌐 Multilingual interface
* 📊 Prediction history
* 🗄️ Database integration

## 👩‍💻 Project

**FreshScan AI — AI Food Quality Analyzer**

Built using deep learning and transfer learning with **MobileNetV2**.

### 🌐 Live Application

**https://freshscan-ai.streamlit.app/**
