# Image-Classification-Using-KNN-with-Histogram-and-Custom-Features
A Python-based image classification system that uses histogram and custom contour-based features to classify images into "cloudy", "shine", or "sunrise" categories using KNN. Features are extracted, saved, and evaluated with training, validation, and testing phases.

# NOTE
Can't upload dataset.zip because file size limitations. Will provide upon request if needed.

# Image Classification Using KNN
This project implements an image classification pipeline for three classes of natural scene images: cloudy, shine, and sunrise. It uses two feature extraction techniques and classifies images with the K-Nearest Neighbors (KNN) algorithm.

# Dataset
The dataset consists of 908 images divided as follows:

Training Set: First 50% of each class

Validation Set: Next 25% of each class

Test Set: Final 25% of each class

# Features
1. Histogram Features
Based on the V (brightness) channel of the HSV color space

32-bin histogram representing brightness distribution

2. Mystery Features (Contour-Based)
Extracted using Canny edge detection and contour analysis

# Features include:

Contour area

Perimeter

Solidity (area-to-convex-hull ratio)

Helps capture shape and structure of visual elements

Classifier
K-Nearest Neighbors (KNN) with Manhattan (L1) distance

training() evaluates multiple K values (1, 3, 5, 7) using validation accuracy and selects the best

testing() evaluates accuracy on the test set using the best K

# Output
Extracted features are saved as .npy files in class-specific folders

Validation and test accuracies are printed

A plot showing validation accuracy for each K value is generated

# Requirements
Python 3.x

NumPy

OpenCV (cv2)

scikit-learn

Matplotlib

# Install dependencies using:

pip install numpy opencv-python scikit-learn matplotlib
