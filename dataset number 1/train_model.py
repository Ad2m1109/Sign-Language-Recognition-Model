import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('archive/sign_mnist_train.csv')
test_df = pd.read_csv('archive/sign_mnist_test.csv')
print("Datasets loaded.")

# Separate features and labels
print("Separating features and labels...")
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
print("Features and labels separated.")

# Normalize pixel values
print("Normalizing pixel values...")
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0
print("Pixel values normalized.")

# Reshape pixel data into image format (28x28x1)
print("Reshaping image data...")
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print("Image data reshaped.")

# One-hot encode labels
print("One-hot encoding labels...")
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)
print("Labels one-hot encoded.")

print("Data preprocessing complete.")

# Define the CNN model
print("Building CNN model...")
model = Sequential([
    Conv2D(75, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(50, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(25, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax') # Output layer with number of classes
])

# Compile the model
print("Compiling model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
print("Training model...")
history = model.fit(X_train, y_train,
                    epochs=10, # You can adjust the number of epochs
                    batch_size=128,
                    validation_data=(X_test, y_test))
print("Model training complete.")

# Evaluate the model
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# You can save the model here if needed
model.save('sign_language_model.h5')

