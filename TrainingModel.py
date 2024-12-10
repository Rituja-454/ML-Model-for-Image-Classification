from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Importing required libraries for model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='Conv2D_1'),
    MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1'),
    Conv2D(64, (3, 3), activation='relu', name='Conv2D_2'),
    MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_2'),
    Flatten(name='Flatten'),
    Dense(512, activation='relu', name='Dense_1'),
    Dropout(0.5, name='Dropout'),
    Dense(10, activation='softmax', name='Output')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train, 
    batch_size=64, 
    epochs=10, 
    validation_data=(x_test, y_test)
)

# Save the model
model.save('cifar10_model.h5')
