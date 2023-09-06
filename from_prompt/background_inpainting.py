from diffusers import StableDiffusionInpaintPipeline
import torch
import requests
from io import BytesIO
import PIL
from PIL import Image
import numpy as np
from cac_v1_5 import *
import cv2
import re
from tensorflow.keras.preprocessing.text import text_to_word_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model_path = "runwayml/stable-diffusion-inpainting"
model_path = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    #revision='fp16',
    torch_dtype=torch.float16,
).to(device)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def remove_articles(text):
   
    pattern = r'\b(a|an|the)\b' #remove article       
    text_without_articles = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text_without_articles

prompt = 'a lake between mountains at sunset'


to_prompt = remove_articles(prompt)
to_prompt = text_to_word_sequence(to_prompt)
print(to_prompt)



prompt = [prompt]

g_cpu = torch.Generator().manual_seed(8888)
controller = AttentionStore()
image, x_t = run_and_display(prompt, controller, latent=None, run_baseline=False, generator=g_cpu) #compare.jpg 저장
correct = 'sunset->midnight sky'
command = correct.split('->')
prev,curr = command[0], command[1]

to_prompt = to_prompt.remove(prev)

for prev in to_prompt:
    w_att_map = show_word_cross_attention(prompt,prev,controller, res=16, from_where=("up", "down")) #word_attention_map.jpg 저장
    w_att_map = ptp_utils.view_images(w_att_map)
    w_att_map.save('word_{}_attention_map.jpg'.format(prev),'JPEG')

mask_concat = np.zeros((512,512))
for prev in to_prompt:
    img = cv2.imread('word_{}_attention_map.jpg'.format(prev),0)
    otsu_threshold, _ = cv2.threshold(img,-1,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(otsu_threshold)

    th, mask_2 = cv2.threshold(img, otsu_threshold, 255, cv2.THRESH_BINARY)
    print(mask_2.shape)

    maskk = Image.fromarray(mask_2)
    maskk.save('word_{}_mask.jpg'.format(prev))


    white = np.where(mask_2 == 255)
    #black = np.where(np.logical_or.reduce(mask_2 == 0))

    mask_concat[white] = 255

print(mask_concat)
print(mask_concat.shape)
output = Image.fromarray(mask_concat).convert('RGB')
output.save('mask_concat.jpg')

w = np.where(mask_concat == 0)
b = np.where(mask_concat == 255)

mask_concat[w] = 255
mask_concat[b] = 0

inversion = Image.fromarray(mask_concat).convert('RGB')
inversion.save('mask_inversion.jpg')

image = PIL.Image.open('source_image.jpg').resize((512,512))
mask_image = PIL.Image.open('mask_inversion.jpg').resize((512,512))

#corrected_prompt = prompt[0].replace(prev,curr)
#print(corrected_prompt)


image = pipe(prompt=curr,image=image, mask_image=mask_image).images[0]
image.save("inpainting_background.jpg")