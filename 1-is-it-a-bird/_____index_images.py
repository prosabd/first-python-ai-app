import _search as search
from fastai.vision.all import *
from time import sleep
from multiprocessing import freeze_support


###### Indexing the images
if __name__ == '__main__':
    # avoid error of mulltiprocessing
    freeze_support()
    
    # Define search items and set path to image directory
    searches = 'forest','bird'
    path = Path('images')

    # Iterate over each search term
    for o in searches:
        # Create destination directory for images
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        
        # Download images for each search term
        download_images(dest, urls=search.search_images(f'{o} photo'))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search.search_images(f'{o} sun photo'))
        sleep(10)
        download_images(dest, urls=search.search_images(f'{o} shade photo'))
        sleep(10)
        
        # Resize images in the directory
        resize_images(path/o, max_size=400, dest=path/o)