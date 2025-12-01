# Assignment 2 Results

## Summary
This assignment explores two core areas of computer vision: Deep Learning for Image Classification and Geometric Computer Vision for Panorama Stitching. I implemented a Convolutional Neural Network (ResNet-18) to classify images from the CIFAR-10 dataset and developed a pipeline to stitch multiple overlapping images into a seamless panorama.

## Tech Stack
*   **Language:** Python
*   **Libraries:**
    *   `torch` (PyTorch): For building, training, and evaluating the deep learning models (ResNet-18).
    *   `torchvision`: For dataset loading (CIFAR-10) and image transformations/augmentation.
    *   `opencv-python` (cv2): For feature detection (SIFT/ORB), descriptor matching, homography estimation, and image warping.
    *   `numpy`: For numerical operations and matrix manipulations.
    *   `matplotlib`: For visualizing images, keypoints, and training metrics.
    *   `tqdm`: For progress bars during model training.

## Tech Challenge
The main technical challenges addressed in this assignment were:
1.  **Deep Learning Optimization:** Training a deep residual network (ResNet-18) requires careful hyperparameter tuning, data augmentation (random crops, flips) to prevent overfitting, and learning rate scheduling (OneCycleLR) for efficient convergence.
2.  **Feature Matching & Robustness:** In panorama stitching, accurately matching features between images is difficult due to noise and repetitive patterns.
3.  **Homography Estimation:** Computing the correct transformation between images in the presence of incorrect matches (outliers).
4.  **Seamless Blending:** Aligning and combining images without visible seams or artifacts.

## Solution
To address these challenges, the following solutions were implemented:
*   **Image Classification (`train_cifar10.py`, `image_classification.ipynb`):**
    *   **Architecture:** Utilized a ResNet-18 architecture, known for its ability to train deep networks effectively using skip connections.
    *   **Training Pipeline:** Implemented a robust training loop with `CrossEntropyLoss`, `SGD` optimizer with momentum, weight decay, and `OneCycleLR` scheduler.
    *   **Data Augmentation:** Applied normalization, random horizontal flips, and random crops to improve model generalization.
*   **Panorama Stitching (`panorama_stitching.ipynb`):**
    *   **Feature Detection:** Used robust feature detectors (like SIFT/ORB) to identify distinctive keypoints in overlapping images.
    *   **Matching:** Employed descriptor matchers (e.g., Brute-Force or FLANN) to find correspondences.
    *   **RANSAC:** Applied Random Sample Consensus (RANSAC) to robustly estimate the Homography matrix, effectively filtering out outlier matches.
    *   **Warping & Blending:** Used the computed homography to warp images into a common coordinate system and blended them to create the final panorama.

## Impact
The completed assignment demonstrates:
*   **High-Performance Classification:** The ResNet-18 model achieves competitive accuracy on the CIFAR-10 dataset, validating the effectiveness of modern CNN architectures and training techniques.
*   **Practical Computer Vision:** The panorama stitching pipeline successfully creates wide-angle views from standard photos, showcasing the practical application of geometric transformations and feature matching.
*   **Reproducibility:** The provided notebooks and scripts allow for easy reproduction of results and further experimentation with different models or datasets.
