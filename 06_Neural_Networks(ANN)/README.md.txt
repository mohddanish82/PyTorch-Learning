# Wine Classification Using PyTorch Neural Network

This project demonstrates how to build and train a simple neural network using PyTorch on the Wine Dataset.

## Dataset

The dataset contains 13 numerical features describing different chemical properties of wine.

Target classes:
- Class 1
- Class 2
- Class 3

## Neural Network Architecture

Input Layer:
- 13 Features

Hidden Layers:
- Linear(13, 64)
- ReLU
- Linear(64, 32)
- ReLU

Output Layer:
- Linear(32, 3)

## Loss Function

- CrossEntropyLoss

## Optimizer

- Adam
- Learning Rate: 0.001

## Steps

1. Load dataset
2. Split features and labels
3. Normalize features
4. Create train and test sets
5. Build neural network
6. Train model
7. Evaluate accuracy
8. Make predictions

## Requirements

```bash
pip install torch pandas scikit-learn