# Weather Detection

This project uses deep learning to classify weather conditions from images using TensorFlow and Keras. It leverages the InceptionV3 model for transfer learning and applies data augmentation for improved model performance.

---

## Features
- **Data Preprocessing**: Automatic rescaling and augmentation of image data.
- **Model Architecture**: InceptionV3 model for feature extraction and weather classification.
- **Training and Validation**: Trains the model on a labeled dataset and evaluates it on validation data.
- **Random Image Prediction**: Tests the trained model on a random image from the validation dataset.
---

## Setup and Requirements

### Dependencies
Python 3.7+
TensorFlow 2.18+
NumPy
Pandas

## Dataset
The dataset is not included. Please download it from https://www.kaggle.com/datasets/jehanbhathena/weather-dataset and split it into training, testing, and validation.

Install the required packages:
```bash
pip install tensorflow numpy pandas
