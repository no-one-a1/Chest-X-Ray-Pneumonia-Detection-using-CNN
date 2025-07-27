# Chest X-Ray Pneumonia Detection using CNN

## 🩻 Overview

This project focuses on building a Convolutional Neural Network (CNN) model to classify chest X-ray images as either **Normal** or **Pneumonia**. It uses deep learning techniques to help detect pneumonia from medical imaging data, providing an assistive tool for healthcare professionals.

## 📁 Dataset

The dataset used is the **Chest X-ray Images (Pneumonia)** dataset, typically structured in three folders:
- `train/`
- `val/`
- `test/`

Each folder contains images classified into:
- `NORMAL`
- `PNEUMONIA`

Dataset Source: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 🧠 Model Architecture

The model is built using **TensorFlow/Keras** and includes:
- Convolutional layers (`Conv2D`)
- Pooling layers (`MaxPooling2D`)
- Batch Normalization
- Dense fully connected layers
- Dropout regularization
- Softmax output for binary classification

### Libraries Used
- Python
- TensorFlow / Keras
- OpenCV
- PIL
- Pandas, Seaborn, Matplotlib (for EDA and visualization)
- Scikit-learn (for splitting and metrics)

## 📊 Evaluation

The model's performance is evaluated using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

## 🚀 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/chest-xray-pneumonia-cnn.git
    cd chest-xray-pneumonia-cnn
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook:
    ```bash
    jupyter notebook src_(1)[1].ipynb
    ```

4. Make sure to place the dataset inside a folder named `chest_xray/` in the project root.

## 📌 Results

The trained model achieves high accuracy in distinguishing between healthy and pneumonia-affected lungs. Confusion matrices and metrics are included in the notebook for reference.

