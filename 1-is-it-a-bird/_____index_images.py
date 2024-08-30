import _search as search
from fastai.vision.all import *
from multiprocessing import freeze_support


###### Indexing the images
if __name__ == '__main__':
    freeze_support()
    
    searches = 'forest','bird'
    path = Path('images')
    from time import sleep

    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search.search_images(f'{o} photo'))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search.search_images(f'{o} sun photo'))
        sleep(10)
        download_images(dest, urls=search.search_images(f'{o} shade photo'))
        # sleep(10)
        resize_images(path/o, max_size=400, dest=path/o)