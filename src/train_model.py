"""
Script for training and saving a handwritten digit recognition model using MNIST dataset.

This module handles:
- Data loading and preprocessing
- Neural network model definition
- Model training and validation
- Model persistence
"""

import os
from typing import Tuple
import numpy as np
import tensorflow as tf

# ------------------------ Data Loading & Preprocessing ------------------------
def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                       Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset.
    
    Returns:
        Tuple containing:
        - (x_train, y_train): Training data and labels
        - (x_test, y_test): Testing data and labels
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    return (x_train, y_train), (x_test, y_test)

# ------------------------ Model Definition ------------------------
def create_model() -> tf.keras.models.Sequential:
    """Create a sequential neural network model for digit classification.
    
    Architecture:
    - Input layer: Flatten 28x28 images to 784 pixels
    - Hidden layers: Two dense layers with 128 units and ReLU activation
    - Output layer: 10 units with softmax activation for class probabilities
    
    Returns:
        Compiled TensorFlow/Keras model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # First hidden layer
        tf.keras.layers.Dense(128, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Suitable for integer labels
        metrics=['accuracy']
    )
    
    return model

# ------------------------ Main Execution ------------------------
if __name__ == "__main__":
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create and train model
    model = create_model()
    print("Model architecture summary:")
    model.summary()
    
    print("\nStarting training...")
    model.fit(
        x_train, 
        y_train, 
        epochs=10,
        validation_data=(x_test, y_test)  # Added validation for better monitoring
    )
    
    # Ensure directory exists before saving
    os.makedirs('models', exist_ok=True)
    model.save('models/handwritten-model.keras')
    print("\nModel saved successfully at models/handwritten-model.keras")