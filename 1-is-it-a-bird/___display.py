import __download as download

###### Displaying the image of bird 
from fastai.vision.all import *

im = Image.open(download.dest)
im.to_thumb(256,256)
