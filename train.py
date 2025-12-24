
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Import all dataset classes to ensure they are registered ---
from datasets import DatasetRegistry
from sign_mnist_dataset import SignMnistDataset
from ardamavi_dataset import ArdaMaviDataset
from indian_sign_language_dataset import IndianSignLanguageDataset
from npy_dataset import NpyDataset

# --- Constants ---
MODEL_SAVE_DIR = "trained_models"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128

def get_dataset_data_dir(dataset_name):
    """Returns the relative data directory for a given dataset."""
    dataset_map = {
        "SignMnistDataset": "dataset number 1/archive",
        "NpyDataset": "dataset number 2/archive/Sign-language-digits-dataset",
        "ArdaMaviDataset": "dataset number 3/archive",
        "IndianSignLanguageDataset": "dataset number 4/archive/ISL_Dataset"
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Data directory for dataset '{dataset_name}' is not defined.")
    return dataset_map[dataset_name]

def train(dataset_name, epochs, batch_size, model_filename):
    """
    Main training function.
    """
    # --- Setup ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    full_model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    data_dir = get_dataset_data_dir(dataset_name)

    # --- Data Loading and Preprocessing ---
    print(f"Loading dataset: {dataset_name} from {data_dir}...")
    try:
        dataset_instance = DatasetRegistry.get_dataset(dataset_name)
        dataset_instance.data_dir = data_dir
        dataset_instance.load_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    X_train = dataset_instance.X_train
    y_train = dataset_instance.y_train
    X_test = dataset_instance.X_test
    y_test = dataset_instance.y_test
    num_classes = dataset_instance.get_dataset_info()['num_classes']

    print("Data preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"Number of classes: {num_classes}")

    # --- Define the CNN model ---
    print("Building CNN model...")
    # This is a generic model architecture; it may need tuning for different datasets
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
        Dense(num_classes, activation='softmax')
    ])

    # --- Compile the model ---
    print("Compiling model...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # --- Train the model ---
    print(f"Training model for {epochs} epochs...")
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test))
    print("Model training complete.")

    # --- Evaluate the model ---
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Save the model ---
    print(f"Saving model to {full_model_path}...")
    model.save(full_model_path)
    print(f"Model saved successfully to {full_model_path}")

if __name__ == '__main__':
    available_datasets = DatasetRegistry.get_available_datasets()
    
    parser = argparse.ArgumentParser(description="A unified training script for the Sign Language Recognition Framework.")
    parser.add_argument('--dataset', type=str, required=True, choices=available_datasets,
                        help=f"The name of the dataset to train on. Available: {', '.join(available_datasets)}")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs. Default: {DEFAULT_EPOCHS}")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training. Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument('--output-model-name', type=str, required=True,
                        help="The filename for the saved model (e.g., 'sign_mnist_v1.h5').")

    args = parser.parse_args()

    train(args.dataset, args.epochs, args.batch_size, args.output_model_name)
