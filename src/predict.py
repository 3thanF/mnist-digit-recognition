import numpy as np
import tensorflow as tf

class Prediction:
    def __init__(self, model_path='../models/mnist_cnn.h5'):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, image):
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        image = image.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return image
    
    def predict(self, image):
        processed_image = self.preprocess(image)
        prediction = self.model.predict(processed_image)
        return {
            'digit': int(np.argmax(prediction)),
            'confidence': float(np.max(prediction)),
            'probabilities': prediction.tolist()[0]
        }