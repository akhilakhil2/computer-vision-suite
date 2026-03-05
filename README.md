# computer-vision-suite
Airbus Aircraft Detection with YOLOv8
Project Overview

This repository contains a Jupyter Notebook (airbus-detection1.ipynb) that demonstrates how to:

    Download the Airbus Aircrafts Sample Dataset from Kaggle.

    Preprocess the satellite imagery and convert bounding box annotations to the YOLO format.

    Set up a structured dataset for training and validation.

    Train a YOLOv8n (Nano) model for object detection.

    Perform inference on test images and visualize the results.

Dataset

The project utilizes the Airbus Aircrafts Sample Dataset.

    Source: Kaggle - Airbus Aircrafts Sample Dataset

    Description: High-resolution satellite images containing various aircraft for object detection tasks.

Key Features

    Automated Dataset Setup: Uses kagglehub to fetch the latest version of the dataset directly into the environment.

    YOLO Formatting: Automatically converts standard CSV bounding box annotations into normalized YOLO .txt files.

    Data Splitting: Implements a standard 80/20 train-validation split.

    Efficient Training: Leverages the ultralytics YOLOv8 framework for fast and accurate training on GPU-enabled environments.

    Visualization: Built-in tools to display raw dataset samples and final model predictions with bounding boxes.

Requirements

To run this project, you need the following libraries installed:
Bash

pip install ultralytics kagglehub pandas matplotlib pillow torch

How to Use

    Clone the Repository:
    Bash

    git clone https://github.com/your-username/airbus-aircraft-detection.git
    cd airbus-aircraft-detection

    Run the Notebook:
    Open airbus-detection1.ipynb in Google Colab or a local Jupyter environment and run the cells sequentially.

    Training:
    The notebook is configured to train for 10 epochs with an image size of 640. You can modify the epochs parameter in the model.train() call for better results.

Results

After training, the model's weights are saved in the runs/detect/train/weights/ directory. The notebook includes a visualization section that uses these weights to predict and display aircraft locations in new images.
