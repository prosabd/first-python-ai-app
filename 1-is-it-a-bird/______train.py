from fastai.vision.all import *
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    path = Path('images')
    
    ## Verify images
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)

    ## Split data
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128)
    ).dataloaders(path, bs=32)

    dls.show_batch(max_n=6)
    
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    
    # Save the trained model
    learn.export('model.pkl')