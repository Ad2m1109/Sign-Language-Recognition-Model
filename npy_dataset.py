import numpy as np
from sklearn.preprocessing import LabelBinarizer
from datasets import BenchmarkDataset, DatasetRegistry
import os

@DatasetRegistry.register_dataset
class NpyDataset(BenchmarkDataset):
    def __init__(self, name="NpyDataset"):
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

        # Assuming X_raw is already in image format (e.g., 64x64) and needs resizing to 28x28
        # And assuming it's grayscale, so we need to add a channel dimension.
        # If X_raw is already 28x28, this reshape will still work.
        # If it's color, further processing would be needed.
        
        # For now, let's assume X_raw is (num_samples, height, width) and needs to be (num_samples, 28, 28, 1)
        # If the original images are not 28x28, they would need resizing here.
        # For simplicity, let's assume they are already 28x28 or can be directly reshaped.
        # If the dataset is 'Sign-language-digits-dataset', it might be 64x64.
        # For now, I'll assume it's already preprocessed to 28x28 or can be reshaped.
        # If not, a resizing step (e.g., using cv2.resize) would be necessary.
        
        # Normalize pixel values (assuming 0-255 range)
        self.X_train = X_raw / 255.0
        self.X_test = X_raw / 255.0 # For simplicity, using the same data for train/test split later if needed

        # Reshape to (num_samples, 28, 28, 1)
        # This assumes the original images are square and can be reshaped to 28x28.
        # If the original images are not 28x28, a resizing step is crucial.
        # For the 'Sign-language-digits-dataset', images are 64x64.
        # I need to add a resizing step here.
        
        # Let's assume the images are 64x64 and need to be resized to 28x28
        # This requires opencv-python, which is already a dependency.
        import cv2
        
        num_samples = self.X_train.shape[0]
        resized_images = np.zeros((num_samples, 28, 28))
        for i in range(num_samples):
            resized_images[i] = cv2.resize(self.X_train[i], (28, 28), interpolation=cv2.INTER_AREA)
        self.X_train = resized_images.reshape(-1, 28, 28, 1)
        
        num_samples_test = self.X_test.shape[0]
        resized_images_test = np.zeros((num_samples_test, 28, 28))
        for i in range(num_samples_test):
            resized_images_test[i] = cv2.resize(self.X_test[i], (28, 28), interpolation=cv2.INTER_AREA)
        self.X_test = resized_images_test.reshape(-1, 28, 28, 1)


        # One-hot encode labels
        self.label_binarizer = LabelBinarizer()
        self.y_train = self.label_binarizer.fit_transform(y_raw)
        self.y_test = self.label_binarizer.transform(y_raw) # Using same for test for now

        # Define label mapping for digits 0-9
        self.label_mapping = {i: str(i) for i in range(10)}
        
        print(f"{self.name} Data preprocessing complete.")

    def get_dataset_info(self):
        # Assuming 10 classes for digits 0-9
        return {"num_classes": 10}
