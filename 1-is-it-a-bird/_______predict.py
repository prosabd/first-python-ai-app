import ______train as train
from fastai.vision.all import *

# Load the trained model
learn = load_learner('model.pkl')

is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")