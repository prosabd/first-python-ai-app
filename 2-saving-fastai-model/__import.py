from fastai.vision.all import *
import gradio as gr

# This function checks if the first character of the filename is uppercase
def is_cat(x): return x[0].isupper()

# Predict on one image with previous model
 
## Get the dog picture i downlaoded and convert to PIlImage format
im = PILImage.create('images/dog.jpg')
im.thumbnail((192,192))
im

## Load the trained model and predict on the image
learn = load_learner('model.pkl')
learn.predict(im)

## Define 2 categories for the model to predict on
categories = ('Dog', 'Cat')

## Define a function that takes an image and returns a dictionary of probabilities for each category
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

## Test the function
classify_image(im)

# Try use Gradio interface with the saved model
## Define args for elements to the interface
image = gr.Image(width=192, height=192)
label = gr.Label()
examples = ['images/dog.jpg','images/cat.jpg', 'images/dunno.jpg']

## Create the interface and lanch it
intf = gr.Interface(fn=classify_image, inputs=image, outputs=[label], examples=examples)
intf.launch(inline=False)