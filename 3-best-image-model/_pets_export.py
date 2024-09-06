from fastai.vision.all import *
from multiprocessing import freeze_support
import timm

path = untar_data(URLs.PETS)/'images'


if __name__ == '__main__':
        # avoid error of mulltiprocessing
        freeze_support()        

        dls = ImageDataLoaders.from_name_func('.',
                get_image_files(path), valid_pct=0.2, seed=42,
                label_func=RegexLabeller(r'^([^/]+)_\d+'),
                item_tfms=Resize(224))

        dls.show_batch(max_n=4)

        learn = vision_learner(dls, resnet34, metrics=error_rate)
        learn.fine_tune(4)

        timm.list_models('convnext*')

        # use convnext\_tiny\_in22k model, better model of previous project exported model
        learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error_rate).to_fp16()
        learn.fine_tune(3)

        learn.export('model.pkl')