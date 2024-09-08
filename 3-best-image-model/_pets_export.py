from fastai.vision.all import *
import timm

# inform the user of the availability of NVIDA GPU and MPS (Mac M1,M2,M3 framework of pytorch, like Cuda for nvidia)
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"Nom du GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun'}")
print(f"MPS disponible: {torch.backends.mps.is_available()}")


# download the pets dataset
path = untar_data(URLs.PETS)/'images'      

# Create DataLoader for efficient batching and preprocessing of images
dls = ImageDataLoaders.from_name_func('.',
        get_image_files(path), valid_pct=0.2, seed=42,
        label_func=RegexLabeller(r'^([^/]+)_\d+'),
        item_tfms=Resize(224))

# 'Display' a sample batch of images and their labels
dls.show_batch(max_n=4)

# re-check if mps is available (Mac M1,M2,M3 framework of pytorch, like Cuda for nvidia)
if torch.backends.mps.is_available():
        device = torch.device("mps")
elif torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")
print(f"Using device: {device}")

# Create and train the first model using ResNet34 architecture
learn = vision_learner(dls, 'resnet34', metrics=error_rate)
learn.to(device)  # Move the model to the appropriate device
learn.fine_tune(4)  # Fine-tune the model for 4 epochs

# List available ConvNeXt models from the timm library
timm.list_models('convnext*')

# Create and train the second model using ConvNeXt Tiny architecture, heavier model of previous project exported model (project 2)
learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error_rate).to_fp16()
learn.to(device)  # Move the model to the appropriate device
if device.type == 'cuda':
        learn.to_fp16()  # Use mixed precision training for CUDA devices to speed up training
learn.fine_tune(3)  # Fine-tune the model for 3 epochs

learn.export('model.pkl')