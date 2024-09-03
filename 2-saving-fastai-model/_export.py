from fastai.vision.all import *
from multiprocessing import freeze_support

# Set path to the image directory
path = untar_data(URLs.PETS)/'images'

# Define a function to determine if an image is a cat or not, with a check if the first character of the filename is uppercase
def is_cat(x): return x[0].isupper()

if __name__ == '__main__':
    # avoid error of mulltiprocessing
    freeze_support()
        

    # Create ImageDataLoaders and use the from_name_func method, from the image files, to create the loader of the actual path '.'
    ## method designed to load image data and prepare it for training a machine learning model
    dls = ImageDataLoaders.from_name_func('.',
        get_image_files(path), # Get all image files in the path
        valid_pct=0.2, # Use 20% of the data for validation
        seed=42, # Set a seed for reproducibility
        label_func=is_cat, # Label function to determine if an image is a cat or not
        item_tfms=Resize(192) # Resize all images to 192x192
    )

    # Create a vision learner with resnet18 model and fine-tune it (for 3 epochs)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    # Export the trained model to a file
    learn.export('model.pkl')