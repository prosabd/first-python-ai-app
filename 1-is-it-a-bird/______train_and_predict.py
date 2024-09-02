from fastai.vision.all import *
from multiprocessing import freeze_support

if __name__ == '__main__':
    # avoid error of mulltiprocessing
    freeze_support()
    
    # Set path to image directory
    path = Path('images')
    
    # Verify images in the directory
    ## Check for corrupted or invalid images
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)

    # Split data into training and validation sets
    ## Create DataBlock (with ImageBlock and CategoryBlock), a method designed to data preparation for machine learning models.
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128)
    ).dataloaders(path, bs=32)

    # Display a batch of images
    dls.show_batch(max_n=6)
    
    # Create a vision learner with resnet18 model & Fine-tune it (for 4 epochs)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    ## Test predict with first download of bird
    is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")