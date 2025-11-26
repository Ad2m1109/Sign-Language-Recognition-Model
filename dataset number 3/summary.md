# Dataset 3: ArdaMavi Sign Language Digits

## Overview
This dataset contains images of sign language digits (0-9). It is sourced from the ArdaMavi dataset.

## Structure
-   `archive/`: Contains `X.npy` (images) and `Y.npy` (labels).
-   `train.py`: Script to train the CNN model.
-   `deploy.py`: Script to run real-time inference using the trained model.
-   `trained_models/`: Contains the saved model `sign_language_model_ardamavi.h5`.

## Training
The model is a Convolutional Neural Network (CNN) trained on 64x64 images resized to 28x28 grayscale.
-   **Input Shape**: (28, 28, 1)
-   **Classes**: 10 (Digits 0-9)
-   **Optimizer**: Adam
-   **Loss**: Categorical Crossentropy

## Usage
To use this dataset:
1.  Train the model: `python3 train.py`
2.  Run inference: `python3 deploy.py`
    -   Or use the main interface: `python3 ../main.py` and select Option 3.
