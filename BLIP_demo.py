#%%
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import transformers
import cv2
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

raw_image = Image.open('./redcar.jpg')

inputs = processor(raw_image, return_tensors="pt").to(device)

out = model.generate(**inputs)

#raw_image.show()

print('Caption :',processor.decode(out[0], skip_special_tokens=True))
# %%
