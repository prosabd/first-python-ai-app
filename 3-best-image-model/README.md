# 3 Best Image Models

This repository contains a collection of AI projects focused on image classification and model benchmarking using PyTorch and Fastai. The goal is to provide a comprehensive set of tools and resources for training, validating, and deploying state-of-the-art image classification models.

## Overview

The `3-best-image-model` directory includes various projects that demonstrate different aspects of image classification, model training, and evaluation. Each project is designed to be self-contained, allowing users to easily understand and replicate the results.

## Project Structure

- **_pets_export.py**: This script is responsible for training a Fastai model using a dataset of pet images (cats and dogs). It defines the data loading process, model architecture, and training procedure. After training, it exports the model to a file named `model.pkl` for later use.

- **__read_data.py**: This script loads and processes benchmark data from the PyTorch Image Models repository. It merges this data with ImageNet results, allowing for detailed analysis of model performance across different architectures and GPUs.

- **___display_data.py**: Similar to the previous script, this one displays overall model performance data, allowing users to visualize the results of different models in a user-friendly format.

- **____display_sub_data.py**: This script utilizes Plotly to visualize the performance of various models, focusing on pretrained ConvNeXt models. It provides scatter plots to analyze inference times and accuracy metrics.

## How It Works

1. **Data Preparation**: The `_pets_export.py` script downloads a dataset of pet images and prepares it for training. It splits the data into training and validation sets.

2. **Model Training**: A convolutional neural network (CNN) is created using the ResNet architecture. The model is trained for a specified number of epochs to minimize classification error.

3. **Model Export**: After training, the model is saved to a file named `model.pkl`, which can be loaded later for inference.

4. **Data Analysis**: The `__read_data.py` script loads benchmark results from the PyTorch Image Models repository, allowing for a comprehensive analysis of model performance across various architectures and GPUs.

5. **Visualization**: The `____display_sub_data.py` and `___display_data.py` scripts use Plotly to create interactive visualizations of model performance, helping users understand the efficiency and accuracy of different models.
