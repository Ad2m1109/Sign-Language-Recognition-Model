# Sign Language Recognition Model

This project is a sign language recognition system with **hand detection** that can identify alphabet characters (A-Z) and numbers (0-9) from live camera feed. It includes 4 different datasets for comparison and benchmarking.

## Key Features

- **Hand Detection**: Automatic hand detection using OpenCV skin color detection
- **4 Datasets**: Compare performance across different sign language datasets
- **Real-time Recognition**: Live camera feed with instant predictions
- **Centralized Interface**: Easy-to-use menu to test all models
- **Confidence Scores**: See how confident the model is in its predictions

## Project Structure

- `main.py`: **Central interface** - Run this to select and test any dataset model
- `inference.py`: **Shared inference logic** with hand detection for all models
- `datasets.py`: Defines the `BenchmarkDataset` abstract base class and the `DatasetRegistry` for managing multiple datasets
- `sign_mnist_dataset.py`: Implements the `SignMnistDataset` for alphabet recognition
- `npy_dataset.py`: Implements the `NpyDataset` for digit recognition from .npy files
- `ardamavi_dataset.py`: Implements the `ArdaMaviDataset` for digit recognition
- `indian_sign_language_dataset.py`: Implements the `IndianSignLanguageDataset` for alphabet recognition
- `requirements.txt`: All Python dependencies
- `README.md`: This file
- `trained_models/`: Contains all trained models
    - `sign_language_model_sign_mnist.h5` (Dataset 1)
    - `sign_language_model_digits.h5` (Dataset 2)
    - `sign_language_model_ardamavi.h5` (Dataset 3)
    - `sign_language_model_indian.h5` (Dataset 4)
- `dataset number 1/`: Sign MNIST dataset (Alphabet A-Z)
    - `archive/`: Contains the raw dataset files (`sign_mnist_train.csv`, `sign_mnist_test.csv`)
    - `train.py`: Script to train the model for this dataset
    - `deploy.py`: Script to deploy the trained model for real-time prediction
    - `summary.md`: Detailed outline of the model training steps
- `dataset number 2/`: NPY-based sign language digits dataset (0-9)
    - `archive/Sign-language-digits-dataset/`: Contains the raw dataset files (`X.npy`, `Y.npy`)
    - `train.py`: Script to train the model for this dataset
    - `deploy.py`: Script to deploy the trained model for real-time prediction
    - `summary.md`: Detailed outline of the model training steps
- `dataset number 3/`: ArdaMavi sign language digits dataset (0-9)
    - `archive/`: Contains the raw dataset files (`X.npy`, `Y.npy`)
    - `train.py`: Script to train the model for this dataset
    - `deploy.py`: Script to deploy the trained model for real-time prediction
    - `summary.md`: Detailed outline of the model training steps
- `dataset number 4/`: Indian Sign Language dataset (Alphabet, 23 classes)
    - `archive/ISL_Dataset/`: Contains subdirectories for each letter class with images
    - `train.py`: Script to train the model for this dataset
    - `deploy.py`: Script to deploy the trained model for real-time prediction
    - `summary.md`: Detailed outline of the model training steps


## Dataset Management

This project uses a `DatasetRegistry` to manage various `BenchmarkDataset` implementations. The core dataset logic (`datasets.py`, `sign_mnist_dataset.py`, `npy_dataset.py`) remains in the root directory to be shared by all dataset-specific scripts.

### Current Datasets

1. **SignMnistDataset** (Dataset 1): Alphabet A-Z (24 classes, excluding J and Z). Located in `dataset number 1/`.
2. **NpyDataset** (Dataset 2): Digits 0-9. Located in `dataset number 2/`.
3. **ArdaMaviDataset** (Dataset 3): Digits 0-9 from ArdaMavi. Located in `dataset number 3/`.
4. **IndianSignLanguageDataset** (Dataset 4): Alphabet (23 classes) from Indian Sign Language. Located in `dataset number 4/`.

### Adding New Datasets

To add a new dataset:
1.  Create a new directory for your dataset (e.g., `my_new_dataset/`).
2.  Place your raw data files within `my_new_dataset/archive/`.
3.  Copy `train.py`, `deploy.py`, and `summary.md` from an existing dataset folder (e.g., `dataset number 1/`) into `my_new_dataset/`.
4.  Create a `trained_models/` directory inside `my_new_dataset/`.
5.  Create a new Python file (e.g., `my_new_dataset_impl.py`) that defines a class inheriting from `BenchmarkDataset` (from `datasets.py`). This file should be placed in the root directory.
6.  Implement the `load_data()` method within your new class to handle loading, preprocessing, reshaping, normalizing, and one-hot encoding of your dataset's data. Ensure `X_train`, `y_train`, `X_test`, `y_test`, and `label_mapping` attributes are populated. The `data_dir` attribute of your dataset instance will be set by the `train.py` and `deploy.py` scripts within `my_new_dataset/`.
7.  Decorate your new dataset class with `@DatasetRegistry.register_dataset` to make it available to the system.
8.  Ensure your new dataset implementation file (e.g., `my_new_dataset_impl.py`) is imported in the `train.py` and `deploy.py` scripts within `my_new_dataset/` (and any other relevant scripts) so it gets registered.
9.  Modify the `train.py` and `deploy.py` scripts within `my_new_dataset/` to hardcode the `SELECTED_DATASET`, `DATA_DIR`, `MODEL_FILENAME`, and `MODEL_SAVE_DIR` variables to be specific to your new dataset.

## Getting Started

To get started with this project, follow the steps below.

### Prerequisites

*   Python 3.x
*   Jupyter Notebook or a Python IDE
*   Required Python libraries (will be installed in the setup)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd sign-language-model
    ```
    (Note: Replace `<repository_url>` with the actual URL if this project is hosted on Git.)

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy tensorflow keras scikit-learn matplotlib opencv-python
    ```

## Usage

### Quick Start (Recommended)

Run the central interface to test any model:
```bash
python3 main.py
```

Select a dataset (1-4) from the menu and the system will:
- Load the trained model
- Initialize your camera
- **Detect your hand** using skin color detection
- Show real-time predictions with confidence scores
- Display a green bounding box around your hand

Press 'q' to quit the camera view.

### Hand Detection Tips

For best results:
- Ensure good lighting
- Use a plain background (not skin-colored)
- Keep your hand centered in the frame
- Wear long sleeves to reduce false detections
- Make clear, distinct gestures

### Training a Model

To train a model for a specific dataset:
```bash
cd "dataset number X"  # Replace X with 1, 2, 3, or 4
python3 train.py
```

The trained model will be saved to `../trained_models/`.

### Individual Dataset Testing

To run inference for a specific dataset without the menu:
```bash
cd "dataset number X"
python3 deploy.py
```


## Training Steps (High-Level)

The training process involves the following key stages:

1.  **Data Loading and Preprocessing:** Handled by the selected `BenchmarkDataset` implementation (e.g., `SignMnistDataset` or `NpyDataset`).
2.  **Model Architecture:** A Convolutional Neural Network (CNN) is defined in the respective `train.py` script.
3.  **Model Compilation:** Configured with an optimizer, loss function, and metrics.
4.  **Model Training:** The CNN is trained on the preprocessed data.
5.  **Model Evaluation:** Model performance is assessed on the test set.
6.  **Deployment with Camera Integration:** The trained model is integrated with a live camera feed for real-time predictions.
