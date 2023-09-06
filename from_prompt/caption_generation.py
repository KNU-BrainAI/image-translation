#%%
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import transformers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(transformers.__version__)

def Image_captioning(image):

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    inputs = processor(image, return_tensors="pt").to(device)

    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)

raw_image = Image.open('D:/LG_data/source00024.jpg')

caption = Image_captioning(raw_image)

print(caption)

# %%
