📌 Potato Disease Classification (PyTorch CNN)

This project implements a Convolutional Neural Network from scratch using PyTorch to classify potato leaf images into disease categories.
The goal of the project was to understand the full deep-learning pipeline — data loading, augmentation, CNN design, training loop implementation, and evaluation — without relying on pretrained models.

🔧 Features

Custom CNN built fully from scratch (no transfer learning)

Image preprocessing and augmentation using torchvision.transforms

Dataset loading with ImageFolder

Manual training and evaluation loops in PyTorch

Accuracy tracking across epochs

Reproducible experiments with manual seed

🧠 Model Architecture

3 convolutional blocks

ReLU activations + MaxPooling

Fully connected classifier

CrossEntropy loss with SGD optimizer

Input images are resized to 224×224 before training.

📊 Results

The model achieved approximately 65–70% test accuracy after training for 50 epochs.
This project was built as part of my early deep-learning learning phase to understand CNN fundamentals rather than to maximize benchmark accuracy.

🚀 Tech Stack

Python

PyTorch

Torchvision
