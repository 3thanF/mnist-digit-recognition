# Handwritten Digit Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end system for recognizing handwritten digits using MNIST dataset and convolutional neural networks.

![GIF](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdm85aWJmYmFzbzd4dnlzdnJhbnJraXRsbW8zY2Q4dmF3dWczczhudSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4TtTVTmBoXp8txRU0C/giphy.gif) <!-- Consider adding a demo GIF -->

## Features

- **Model Training**: CNN architecture with 97%+ accuracy
- **Evaluation Suite**: Detailed performance metrics
- **Image Prediction**: Supports batch processing of digit images
- **Preprocessing**: Automatic image normalization and inversion
- **Visualization**: Prediction results with confidence scores

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Prerequisites

- Python 3.8+
- pip package manager
- 1GB+ free disk space

## Installation

1. Clone repository:
```bash
git clone https://github.com/3thanF/digit-recognition.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create necessary directories:
```bash
mkdir -p models digits
```

## Usage
1. Train the Model:
```bash
python train_model.py
```
### Default Settings:
- Epochs: 10
- Batch Size: 32
- Model Saved to: models/handwritten-model.keras

2. Evaluate Model Performance:
```bash
python evaluation.py
```
Sample Output:
```bash
Evaluation Results:
• Accuracy: 97.32%
• Loss: 0.0894
```

3. Predict Digits from Images:

Place test images in digits/ directory named as digit1.png, digit2.png, etc.

Run predictor:
```bash
python predict.py
```

## Project Structure
```bash
.
├── digits/              # Directory for test images
├── models/              # Saved models
├── train_model.py       # Model training script
├── evaluation.py        # Model evaluation script
├── predict.py           # Prediction script
├── requirements.txt     # Dependency list
└── README.md            # This documentation
```