# Pixel Localization with CNN

A PyTorch project that predicts the (x, y) coordinates of a single bright pixel in 50×50 grayscale images using a convolutional neural network.

## What it does

The model takes images with one bright pixel (value 255) and all other pixels at 0, then predicts where that bright pixel is located. It's trained on all 2,500 possible positions in a 50×50 grid.

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch 2.2.2 (CPU version)
- NumPy <2.0
- scikit-learn
- matplotlib
- tqdm

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook prasan_assignment.ipynb
```

The notebook handles everything:
1. Generates synthetic dataset (2,500 images with one bright pixel each)
2. Splits data into train/val/test (80/10/10)
3. Preprocesses and normalizes images
4. Trains a CNN to predict pixel coordinates
5. Evaluates performance and visualizes results

Data files are saved in `synthetic_data/` and `preprocessed_data/` directories.

## Model

Simple CNN architecture:
- 3 convolutional blocks (32, 64, 128 filters)
- Batch normalization and max pooling
- Fully connected layers with dropout
- Outputs 2 values (x, y coordinates)

Training config:
- Loss: MSE
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 25

## Results

The model achieves near-perfect accuracy since the problem is deterministic with a finite set of 2,500 possible samples. Most predictions are exact or within 1 pixel.

## Notes

This is an academic exercise demonstrating a complete ML pipeline. The problem is simpler than real-world vision tasks - there's no noise, occlusions, or multiple objects. A CNN isn't strictly necessary here, but it shows how to build and train models for spatial localization tasks.
