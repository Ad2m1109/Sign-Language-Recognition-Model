import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

# Load the trained model
print("Loading trained model...")
model = load_model('sign_language_model.h5')
print("Model loaded.")

# It's crucial to have the same LabelBinarizer fitted on the training data
# to correctly inverse transform predictions. For deployment, you'd typically
# save and load this binarizer or reconstruct it based on known classes.
# For this example, let's assume we know the classes and can reconstruct it.
# The Sign MNIST dataset has 24 classes (A-Z, excluding J and Z due to motion).
# We need to create a dummy LabelBinarizer and fit it to the expected labels.
# A simple way is to fit it to a range of numbers that represent the labels.
# The original labels are 0-24, mapping to A-Y (excluding J).
# Let's create a list of characters that correspond to the labels.
# This is a placeholder and should be replaced with the actual mapping
# used during training if available.
# For Sign MNIST, the labels are 0-24, where 9 is skipped (J).
# So, 0=A, 1=B, ..., 8=I, 10=K, ..., 24=Y.
# We need to map these back to actual characters.

# Reconstruct LabelBinarizer (assuming original labels were 0-24, skipping 9)
# This is a critical step. If the original LabelBinarizer was saved, load it.
# Otherwise, reconstruct it carefully.
# For Sign MNIST, the labels are 0-24, excluding 9 (J).
# So, the classes are 0, 1, ..., 8, 10, ..., 24.
# Let's create a list of all possible labels (0-24, excluding 9)
all_labels = list(range(25)) # 0-24
# If 'J' (label 9) was excluded from the original dataset, we need to reflect that.
# Assuming the original dataset had 24 unique classes, corresponding to 0-24 excluding 9.
# The LabelBinarizer will map these 24 unique integers to 24-element one-hot vectors.
# To get the inverse mapping, we need to know what integer corresponds to what character.
# For simplicity, let's assume the model predicts indices 0-23, and we map them to A-Z (skipping J).
# This mapping needs to be consistent with how y_train was encoded.

# A more robust way would be to save the label_binarizer object during training.
# For now, let's create a simple mapping based on the common Sign MNIST convention.
# The dataset labels are 0-24, where 9 is skipped (J).
# So, 0=A, 1=B, ..., 8=I, 10=K, ..., 24=Y.
# We need to map the predicted index (0-23) back to a character.
# Let's define the character mapping:
label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}
# Note: The original dataset skips 'J' (label 9) and 'Z' (label 25).
# So, if the model output is 24 classes, it means it's mapping to A-Y excluding J.
# The `y_train.shape[1]` was 24, so this mapping is consistent.

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