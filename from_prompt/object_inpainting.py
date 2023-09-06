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


prompt = 'a cat sitting on a rock'
#prompt2 = 'a photo of a tree branch at blossom'

prompt = [prompt]
#prompt = [translate_to_english(prompt)]
g_cpu = torch.Generator().manual_seed(8888)
controller = AttentionStore()
image, x_t = run_and_display(prompt, controller, latent=None, run_baseline=False, generator=g_cpu) #prompt로 생성된 이미지 compare.jpg 저장
correct = 'cat->dog'
command = correct.split('->')
prev,curr = command[0], command[1]


att_img = show_word_cross_attention(prompt,prev,controller, res=16, from_where=("up", "down")) #prompt와 image의 cross attention map으로 word_attention_map.jpg 저장
att_result = ptp_utils.view_images(att_img)
att_result.save('word_attention_map.jpg','JPEG')



#threshold = 60
#mask = Image.open('word_attention_map.jpg')

#mask_tensor = np.array(mask)

#white = np.where(np.logical_or.reduce(mask_tensor > threshold, axis=2))
#black = np.where(np.logical_or.reduce(mask_tensor < threshold, axis=2))

#mask_tensor[white] = [255,255,255]
#mask_tensor[black] = [0,0,0]

#output = Image.fromarray(mask_tensor)

#output.save('mask.jpg')

img = cv2.imread('word_attention_map.jpg',0)
otsu_threshold, _ = cv2.threshold(img,-1,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(otsu_threshold)

th, mask_2 = cv2.threshold(img, otsu_threshold, 255, cv2.THRESH_BINARY)

maskk = Image.fromarray(mask_2)
maskk.save('mask.jpg')


#img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
#mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

#image = download_image(img_url).resize((512, 512))
#mask_image = download_image(mask_url).resize((512, 512))
image = PIL.Image.open('source_image.jpg').resize((512,512))
mask_image = PIL.Image.open('mask.jpg').resize((512,512))
#image = PIL.Image.open('dog_bench.jpg').resize((512,512))
#mask_image = PIL.Image.open('dogggggg.png').resize((512,512))

#print(np.array(image).shape,np.array(mask_image).shape)
#corrected_prompt = prompt[0].replace(prev,curr)
#print(corrected_prompt)


image = pipe(prompt=curr,image=image, mask_image=mask_image).images[0]
image.save("inpainting.jpg")


#===================variation을 다르게 줘서 한번에 다수 이미지 변환==================================
# guidance_scale=7.5
# num_samples = 11 #홀수
# generator = torch.Generator(device="cuda").manual_seed(0) 

# images = pipe(
#     prompt=corrected_prompt,
#     image=image,
#     mask_image=mask_image,
#     guidance_scale=guidance_scale,
#     generator=generator,
#     num_images_per_prompt=num_samples,
# ).images

# print(images)

# images.insert(0, image)

# image_grid(images, 2, (num_samples//2)+1).save('one-to-many.jpg')
#


