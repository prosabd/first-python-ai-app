import _search as search


###### Downloading the image of bird
from fastdownload import download_url

dest = 'bird.jpg'
download_url(search.urls[0], dest, show_progress=False)

