"""
Model evaluation script for handwritten digit recognition system.

This module provides:
- Test dataset loading and preprocessing
- Model loading verification
- Performance evaluation metrics calculation
- Result reporting

"""

import tensorflow as tf
from typing import Tuple
import numpy as np
import os

# ------------------------ Data Loading ------------------------
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess MNIST test dataset.
    
    Returns:
        Tuple containing:
        - x_test: Normalized test images (shape: 10000, 28, 28)
        - y_test: Test labels (shape: 10000,)
        
    Raises:
        ValueError: If dataset loading fails
    """
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except Exception as e:
        raise ValueError("Failed to load MNIST dataset") from e
    
    # Normalize pixel values to [0, 1] (matches training preprocessing)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    return x_test, y_test

# ------------------------ Model Loading ------------------------
def load_model(model_path: str = 'models/handwritten-model.keras') -> tf.keras.Model:
    """Load trained model from storage.
    
    Args:
        model_path: Path to saved Keras model
        
    Returns:
        Loaded TensorFlow/Keras model
        
    Raises:
        FileNotFoundError: If model file is missing
        ValueError: If model loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")
        
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        raise ValueError("Failed to load model") from e

# ------------------------ Evaluation ------------------------
def evaluate_model(model: tf.keras.Model, 
                  x_test: np.ndarray, 
                  y_test: np.ndarray) -> Tuple[float, float]:
    """Evaluate model performance on test dataset.
    
    Args:
        model: Loaded Keras model
        x_test: Test images
        y_test: Test labels
        
    Returns:
        Tuple containing:
        - loss: Model's loss value
        - accuracy: Model's accuracy percentage
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, accuracy
    except Exception as e:
        raise RuntimeError("Model evaluation failed") from e

# ------------------------ Main Execution ------------------------
if __name__ == "__main__":
    # Load test data and model
    x_test, y_test = load_data()
    model = load_model()
    
    # Perform evaluation
    loss, accuracy = evaluate_model(model, x_test, y_test)
    
    # Format and display results
    print("\nEvaluation Results:")
    print(f"• Accuracy: {accuracy:.2%}")  # Format as percentage with 2 decimals
    print(f"• Loss: {loss:.4f}")
    print("\nInterpretation:")
    print("- Accuracy > 97% indicates excellent performance on MNIST")
    print("- Loss < 0.1 suggests good model calibration")