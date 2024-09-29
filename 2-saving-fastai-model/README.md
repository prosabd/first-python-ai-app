# 2 Saving Fastai Model

This project demonstrates how to save and load a Fastai model for image classification tasks. It utilizes the Fastai library, which is built on top of PyTorch, to create a model that can classify images of cats and dogs based on their filenames.

## Project Structure

- **_export.py**: This script is responsible for training a Fastai model using a dataset of pet images. It defines the data loading process, the model architecture, and the training procedure. After training, it exports the model to a file for later use.
  
- **__import.py**: This script loads the previously trained model and provides functionality to classify new images. It uses Gradio to create a web interface for users to upload images and receive predictions.

- **requirements.txt**: This file lists the necessary Python packages required to run the project, including Fastai and Gradio.

## How It Works

1. **Data Preparation**: The script downloads a dataset of pet images and prepares it for training. It splits the data into training and validation sets.

2. **Model Training**: A convolutional neural network (CNN) is created using the ResNet architecture. The model is trained for a specified number of epochs to minimize the classification error.

3. **Model Export**: After training, the model is saved to a file named `model_wo_freeze.pkl`, which can be loaded later for inference.

4. **Image Classification**: The `__import.py` script loads the saved model and allows users to classify new images. It provides a web interface where users can upload images, and the model predicts whether the image contains a cat or a dog.

## Usage

To use this project, follow these steps:

1. Run the `_export.py` script to train the model and save it.
2. Run the `__import.py` script to start the Gradio interface for image classification.
3. Upload images through the web interface to see the model's predictions.

## Requirements

Make sure to install the required packages listed in `requirements.txt` before running the scripts:

