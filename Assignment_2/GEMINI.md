# Project Overview

This project consists of two main computer vision tasks implemented in Python using Jupyter Notebooks:

1.  **Image Classification:** This task involves classifying images from the CIFAR-10 dataset. The `image_classification.ipynb` notebook explores different feature extraction methods, including a baseline approach and the Histogram of Oriented Gradients (HOG). It then uses classifiers like K-Nearest Neighbors (KNN) and SGDClassifier from `scikit-learn`. The notebook also includes a challenge to achieve higher accuracy, with bonus points for using PyTorch.

2.  **Panorama Stitching:** This task focuses on stitching two images together to create a panorama. The `panorama_stitching.ipynb` notebook guides through the process, which includes SIFT feature extraction, feature matching, homography estimation using RANSAC, and image blending.

The project relies on several key libraries, including `numpy`, `scikit-learn`, `opencv-python`, `matplotlib`, and `torch`.

# Building and Running

## 1. Environment Setup

It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS and Linux
source venv/bin/activate
```

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

The project also requires the CIFAR-10 dataset. The dataset will be automatically downloaded and extracted when you run the `image_classification.ipynb` notebook for the first time.

## 2. Running the Notebooks

To run the main tasks, start the Jupyter Notebook server:

```bash
jupyter notebook
```

This will open a new tab in your web browser. From there, you can navigate to and run the following notebooks:

*   `image_classification.ipynb`: For the image classification task.
*   `panorama_stitching.ipynb`: For the panorama stitching task.

# Project Structure

*   `image_classification.ipynb`: Jupyter Notebook for the image classification task.
*   `panorama_stitching.ipynb`: Jupyter Notebook for the panorama stitching task.
*   `requirements.txt`: A list of Python packages required for the project.
*   `lib/`: A directory containing helper Python modules.
    *   `dataset.py`: Handles downloading and loading the CIFAR-10 dataset.
    *   `feature_extractor.py`: Contains classes for feature extraction, including the baseline, HOG, and ResNet (PyTorch) extractors.
    *   `vis_utils.py`: Provides utility functions for visualizing images and results.
*   `media/`: Contains the images used for panorama stitching.
*   `stitching_stages/`: Contains sample images showing the results at different stages of the panorama stitching process.
*   `cifar-10-batches-py/`: Directory where the CIFAR-10 dataset is stored.
*   `README.md`: The original README file for the project (in Ukrainian).
