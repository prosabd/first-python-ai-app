# Is It a Bird? - Image Classification Project

This project uses fastai to create a simple image classifier that determines whether an image contains a bird or not.

## File Structure and Execution Order

Files in this project are named with leading underscores. The number of underscores indicates the order of execution:

1. `_search.py`: Image search functionality
2. `__download.py`: Downloads sample images
3. `___display.py`: Displays downloaded images
4. `____download_and_display.py`: Combines download and display functions
5. `_____index_images.py`: Downloads and indexes multiple images for training
6. `______train_and_predict.py`: Trains the model and runs a test prediction

Execute the files in this order to set up the dataset, train the model, and make predictions (for last ```train and predict``` especially).
