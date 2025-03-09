"""
Digit prediction script using a pre-trained MNIST classifier.

Features:
- Sequential image processing from a directory
- Robust error handling for image loading
- Model input validation
- Visual feedback with confidence scores
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ------------------------ Model Loading ------------------------
def load_predictor(model_path: str = 'models/handwritten-model.keras') -> tf.keras.Model:
    """Load trained digit classification model.
    
    Args:
        model_path: Path to saved Keras model file
        
    Returns:
        Loaded TensorFlow model
        
    Raises:
        FileNotFoundError: If model file is missing
        ValueError: For invalid model architecture
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")
        
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        raise ValueError("Failed to load model") from e

# ------------------------ Image Processing ------------------------
def preprocess_image(image_path: str) -> np.ndarray:
    """Prepare image for MNIST model input.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Processed image tensor shaped (1, 28, 28)
        
    Raises:
        ValueError: For invalid image formats or sizes
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found")
        
    # Read and validate image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image from {image_path}")
        
    # Resize and invert (MNIST uses white digits on black background)
    img = cv2.resize(img, (28, 28))
    if np.mean(img) > 127:  # Auto-invert dark-on-light images
        img = 255 - img
        
    # Normalize and add batch dimension
    return (img.astype(np.float32) / 255.0).reshape(1, 28, 28)

# ------------------------ Prediction Logic ------------------------
def predict_digit(model: tf.keras.Model, image: np.ndarray) -> Tuple[int, float]:
    """Make prediction on processed image.
    
    Args:
        model: Loaded TF/Keras model
        image: Preprocessed image tensor
        
    Returns:
        Tuple containing:
        - predicted_class: Integer 0-9
        - confidence: Probability between 0-1
    """
    prediction = model.predict(image, verbose=0)
    return np.argmax(prediction), np.max(prediction)

# ------------------------ Visualization ------------------------
def display_prediction(image: np.ndarray, prediction: int, confidence: float):
    """Show image with prediction results.
    
    Args:
        image: Processed image tensor
        prediction: Predicted digit class
        confidence: Model confidence score
    """
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.1%}")
    plt.axis('off')
    plt.show()

# ------------------------ Main Execution ------------------------
if __name__ == "__main__":
    model = load_predictor()
    image_number = 1
    
    print("Starting digit predictions...")
    while True:
        image_path = f"digits/digit{image_number}.png"
        
        if not os.path.isfile(image_path):
            print(f"\nNo more images found. Processed {image_number-1} files.")
            break
            
        try:
            print(f"\nProcessing {image_path}...")
            processed_img = preprocess_image(image_path)
            digit, confidence = predict_digit(model, processed_img)
            print(f"Predicted digit: {digit} with {confidence:.1%} confidence")
            display_prediction(processed_img, digit, confidence)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            
        finally:
            image_number += 1