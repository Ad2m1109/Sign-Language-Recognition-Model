import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from datasets import DatasetRegistry

def run_realtime_prediction(model_path, dataset_name, data_dir):
    """
    Runs real-time prediction using the specified model and dataset.

    Args:
        model_path (str): Path to the trained .h5 model file.
        dataset_name (str): Name of the registered dataset class.
        data_dir (str): Path to the dataset directory (for loading metadata/labels).
    """

    # Load the trained model
    print(f"Loading trained model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = load_model(model_path)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get the dataset instance to retrieve label mapping
    print(f"Retrieving label mapping for {dataset_name}...")
    try:
        dataset_instance = DatasetRegistry.get_dataset(dataset_name)
        # Pass the data directory to the dataset instance
        dataset_instance.data_dir = data_dir
        # Call load_data to populate label_mapping
        # Note: Ideally, we should have a lighter method just for metadata, 
        # but load_data is what we have for now.
        dataset_instance.load_data() 
        label_mapping = dataset_instance.label_mapping
        print("Label mapping retrieved.")
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0) # 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    print("Camera initialized.")

    print("Starting real-time prediction. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the frame
        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to 28x28
            resized_frame = cv2.resize(gray_frame, (28, 28), interpolation=cv2.INTER_AREA)
            # Normalize pixel values
            normalized_frame = resized_frame / 255.0
            # Reshape for model input (add batch and channel dimensions)
            model_input = normalized_frame.reshape(1, 28, 28, 1)

            # Make prediction
            predictions = model.predict(model_input, verbose=0)
            predicted_class_index = np.argmax(predictions)
            
            # Map predicted index to character
            predicted_character = label_mapping.get(predicted_class_index, "Unknown")

            # Display the prediction on the frame
            cv2.putText(frame, f"Prediction: {predicted_character}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error during prediction: {e}")
            break
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")
