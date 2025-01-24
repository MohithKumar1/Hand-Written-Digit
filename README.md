# Hand-Written-Digit

This repository demonstrates the recognition of handwritten digits using machine learning techniques. The project involves loading the dataset, preprocessing, training models, evaluating performance, and visualizing results.

# Features

Preprocessing handwritten digit data for training.
Implementing classification models using machine learning and deep learning.
Model evaluation using metrics like accuracy and confusion matrix.
Visualization of predictions and misclassified samples.

Dependencies
Ensure you have the following Python libraries installed:

tensorflow
keras
numpy
matplotlib
seaborn
scikit-learn

# Dataset

The dataset used in this project is the MNIST handwritten digit dataset. It consists of:

60,000 training images
10,000 test images
Each image is a grayscale image of size 28x28 pixels, representing digits from 0 to 9.

# Code Overview

1. Dataset Loading and Preprocessing
Load the MNIST dataset using keras.datasets.mnist.
Normalize pixel values to the range [0, 1].
Split data into training and testing sets.

2. Model Architecture
Build a neural network using keras.Sequential.
Include layers such as:
Flatten layer for input preprocessing.
Dense layers with ReLU activation.
Output layer with softmax activation.

3. Model Training
Compile the model with:
Loss: categorical_crossentropy
Optimizer: adam
Metrics: accuracy
Train the model on the training dataset.

4. Model Evaluation
Evaluate the trained model on the test dataset.
Compute metrics like accuracy and loss.
Generate a confusion matrix to analyze misclassifications.

5. Visualization
Plot training and validation accuracy/loss over epochs.
Display sample predictions with their true labels and predicted labels.
Highlight misclassified samples.

