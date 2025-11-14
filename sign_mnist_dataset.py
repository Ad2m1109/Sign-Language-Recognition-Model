import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from datasets import BenchmarkDataset, DatasetRegistry
import os

@DatasetRegistry.register_dataset
class SignMnistDataset(BenchmarkDataset):
    def __init__(self, name="SignMnistDataset"):
        super().__init__(name)
        self.data_dir = None # Will be set by the training/deployment script

    def load_data(self):
        if self.data_dir is None:
            raise ValueError("data_dir must be set before calling load_data()")

        train_csv_path = os.path.join(self.data_dir, 'sign_mnist_train.csv')
        test_csv_path = os.path.join(self.data_dir, 'sign_mnist_test.csv')

        print(f"Loading {self.name} datasets from {train_csv_path} and {test_csv_path}...")
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        print(f"{self.name} Datasets loaded.")

        # Separate features and labels
        print(f"Separating features and labels for {self.name}...")
        X_train_raw = train_df.drop('label', axis=1)
        y_train_raw = train_df['label']

        X_test_raw = test_df.drop('label', axis=1)
        y_test_raw = test_df['label']
        print("Features and labels separated.")

        # Normalize pixel values
        print("Normalizing pixel values...")
        self.X_train = X_train_raw.values / 255.0
        self.X_test = X_test_raw.values / 255.0
        print("Pixel values normalized.")

        # Reshape pixel data into image format (28x28x1)
        print("Reshaping image data...")
        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)
        print("Image data reshaped.")

        # One-hot encode labels
        print("One-hot encoding labels...")
        self.label_binarizer = LabelBinarizer()
        self.y_train = self.label_binarizer.fit_transform(y_train_raw)
        self.y_test = self.label_binarizer.transform(y_test_raw)
        print("Labels one-hot encoded.")

        # Define label mapping for display
        all_chars = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) != 'J']
        self.label_mapping = {i: char for i, char in enumerate(all_chars)}
        
        print(f"{self.name} Data preprocessing complete.")
