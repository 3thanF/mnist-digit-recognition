import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

class MNISTClassifier:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2,2)),
            Flatten(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, epochs=10, batch_size=128, validation_split=0.1):
        # Load and preprocess data
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Create models directory if not exists
        os.mkdirs('../../models', exist_ok=True)

        # Callbacks
        checkpoint = ModelCheckpoint(
            '../../models/mnist_cnn.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint]
        )

        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_acc:.4f}")

        return history
    
    def save(self, file_path='../../models/mnist_cnn.h5'):
        self.model.save(file_path)