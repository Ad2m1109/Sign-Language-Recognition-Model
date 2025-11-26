import numpy as np
from sklearn.preprocessing import LabelBinarizer
from datasets import BenchmarkDataset, DatasetRegistry
import os
import cv2

@DatasetRegistry.register_dataset
class ArdaMaviDataset(BenchmarkDataset):
    def __init__(self, name="ArdaMaviDataset"):
        super().__init__(name)
        self.data_dir = None # Will be set by the training/deployment script

    def load_data(self):
        if self.data_dir is None:
            raise ValueError("data_dir must be set before calling load_data()")

        X_path = os.path.join(self.data_dir, 'X.npy')
        Y_path = os.path.join(self.data_dir, 'Y.npy')

        print(f"Loading {self.name} datasets from {X_path} and {Y_path}...")
        X_raw = np.load(X_path)
        y_raw = np.load(Y_path)
        print(f"{self.name} Datasets loaded.")

        # X_raw is (2062, 64, 64). We need to resize to (2062, 28, 28) and add channel dim.
        
        print("Resizing images from 64x64 to 28x28...")
        num_samples = X_raw.shape[0]
        resized_images = np.zeros((num_samples, 28, 28))
        for i in range(num_samples):
            resized_images[i] = cv2.resize(X_raw[i], (28, 28), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension: (num_samples, 28, 28, 1)
        self.X_train = resized_images.reshape(-1, 28, 28, 1)
        # For simplicity, using the same data for test since we don't have a separate test set file
        self.X_test = self.X_train.copy() 

        # Normalize pixel values (assuming they are already 0-1 if loaded from this specific npy, 
        # but let's check. The original dataset X.npy is usually 0-1 float. 
        # If it's 0-255, we divide. Let's assume it's 0-1 based on typical usage of this dataset, 
        # but safe to clip or check max. 
        # Actually, looking at npy_dataset.py, it divided by 255.0. 
        # Let's assume the raw data is 0-1 range if it's the same file.
        # Wait, if I copied the file from dataset 2, I should check what dataset 2 did.
        # Dataset 2 code: self.X_train = X_raw / 255.0
        # This implies X_raw was 0-255.
        # However, the ArdaMavi dataset usually comes as 0-1 floats.
        # Let's stick to what Dataset 2 did since it's the same file.
        # BUT, if the file is already 0-1, dividing by 255 makes it tiny.
        # I'll add a check.
        if np.max(self.X_train) > 1.0:
             self.X_train = self.X_train / 255.0
             self.X_test = self.X_test / 255.0
        
        print("Images resized and normalized.")

        # One-hot encode labels
        # y_raw is usually (2062, 10) one-hot encoded already for this dataset.
        # Let's check shape.
        if y_raw.shape[1] == 10:
            self.y_train = y_raw
            self.y_test = y_raw.copy()
            # We still need a label binarizer for consistency if we were to inverse transform, 
            # but here we can just set it up.
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(np.arange(10)) # Dummy fit
        else:
            # If not one-hot
            self.label_binarizer = LabelBinarizer()
            self.y_train = self.label_binarizer.fit_transform(y_raw)
            self.y_test = self.label_binarizer.transform(y_raw)

        # Define label mapping for digits 0-9
        self.label_mapping = {i: str(i) for i in range(10)}
        
        print(f"{self.name} Data preprocessing complete.")

    def get_dataset_info(self):
        return {"num_classes": 10}
