import _search as search
import __download as download
import ___display as display

###### Same for forest images

download.download_url(search.search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
display.Image.open('forest.jpg').to_thumb(256,256)