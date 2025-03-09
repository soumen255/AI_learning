import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load the MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist

# Split the data into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the input images (28x28)
    layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons
    layers.Dropout(0.2),  # Dropout layer to reduce overfitting
    layers.Dense(10)  # Output layer with 10 neurons (one for each digit)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model with the training data
model.fit(x_train, y_train, epochs=5)

# Evaluate the model with the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
