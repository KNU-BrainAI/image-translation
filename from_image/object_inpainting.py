from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
import torch
from torchvision.ops import box_convert
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline,AutoPipelineForInpainting
import warnings

warnings.filterwarnings('ignore')

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "mkh.jpg"
TEXT_PROMPT = "mike"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)
print(logits)

max_idx = logits.tolist().index(max(logits.tolist()))

print(max_idx)


annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)

#pipe = StableDiffusionGLIGENPipeline.from_pretrained("gligen/diffusers-inpainting-text-box", revision="fp16", torch_dtype=torch.float16).to('cuda')

def generate_masks_with_grounding(image_source, boxes):
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    single_mask = np.zeros_like(image_source)

    print('box들 :',boxes_xyxy)
    label1 = boxes_xyxy[max_idx]
    a,b,c,d = label1
    single_mask[int(b):int(d),int(a):int(c),:] = 255
    # for box in boxes_xyxy:
    #     x0, y0, x1, y1 = box
    #     mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    # return mask
    return single_mask

image_mask = generate_masks_with_grounding(image_source, boxes)


image_source = Image.fromarray(image_source)


annotated_frame = Image.fromarray(annotated_frame)
annotated_frame.save('dino_sd_detection.jpg')

image_mask = Image.fromarray(image_mask)


image_source_for_inpaint = image_source.resize((512,512))

image_source_for_inpaint.save('dino_sd_raw.jpg')
image_mask_for_inpaint = image_mask.resize((512,512))
image_mask_for_inpaint.save('dino_sd_mask.jpg')
num_box = len(boxes)
print(num_box)

xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
xyxy_boxes[:2]



num_box = len(boxes)

prompt ="gun"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)
generator = torch.Generator(device="cuda").manual_seed(10)



image = pipe(
  prompt=prompt,
  image=image_source_for_inpaint,
  mask_image=image_mask_for_inpaint,
  guidance_scale=8.0,
  num_inference_steps=20, 
  strength=0.99,  
  generator=generator,
).images[0]

image.save('dino_sd_xl.jpg')