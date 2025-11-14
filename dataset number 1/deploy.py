import sys
import os

# Add the parent directory to the Python path to allow importing modules from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Import the DatasetRegistry and ensure SignMnistDataset is registered
from datasets import DatasetRegistry
from sign_mnist_dataset import SignMnistDataset # This import registers the dataset

# --- Hardcoded Dataset Configuration for SignMnistDataset ---
SELECTED_DATASET = "SignMnistDataset"
DATA_DIR = "archive" # Relative to the current script's directory (dataset number 1/)
MODEL_FILENAME = "sign_language_model_sign_mnist.h5"
MODEL_SAVE_DIR = "trained_models" # Relative to the current script's directory (dataset number 1/)

FULL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

# Load the trained model
print(f"Loading trained model from {FULL_MODEL_PATH}...")
model = load_model(FULL_MODEL_PATH)
print("Model loaded.")

# Get the dataset instance to retrieve label mapping
print(f"Retrieving label mapping for {SELECTED_DATASET}...")
dataset_instance = DatasetRegistry.get_dataset(SELECTED_DATASET)
# Pass the data directory to the dataset instance
dataset_instance.data_dir = DATA_DIR
# Call load_data to populate label_mapping (only if necessary, could be optimized to just load metadata)
dataset_instance.load_data() 
label_mapping = dataset_instance.label_mapping
print("Label mapping retrieved.")

# Initialize camera
print("Initializing camera...")
cap = cv2.VideoCapture(0) # 0 for default camera
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
print("Camera initialized.")

print("Starting real-time prediction. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized_frame = cv2.resize(gray_frame, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Reshape for model input (add batch and channel dimensions)
    model_input = normalized_frame.reshape(1, 28, 28, 1)

    # Make prediction
    predictions = model.predict(model_input)
    predicted_class_index = np.argmax(predictions)
    
    # Map predicted index to character
    predicted_character = label_mapping.get(predicted_class_index, "Unknown")

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {predicted_character}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application closed.")