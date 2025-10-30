import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

class Preprocessing:
    def __init__(self):
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

    def load_and_prepare(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # Flatten for ANN (28x28 -> 784)
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        print(f"Train shape: {x_train.shape}, Labels: {y_train.shape}")
        print(f"Test shape: {x_test.shape}, Labels: {y_test.shape}")

    def save_model(self, model, path="D:\\DEPI ASSIGN 14\\Trial\\models\\fashion_ann.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"Model saved successfully at {path}")
