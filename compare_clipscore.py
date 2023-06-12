import torch
import clip
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', device)

#naive = ['Naive prompt', 'The red car in the winter forest.high detailed, insane quality, 4k']
#blip = ['BLIP_caption', 'a red car driving through a snowy forest']
#any = ['Any_caption', 'a baby in the house']
#caption_list = [raw,blip,any]

korean_prompt_list = []

def calculate_clip_score(cap,image_path):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    caption = clip.tokenize([cap[1]]).to(device)


    with torch.no_grad():
        image_embedding = model.encode_image(image).float()
        caption_embedding = model.encode_text(caption).float()


    score = 100 * float(torch.cosine_similarity(image_embedding, caption_embedding))

    print('{}과의 CLIP score : {}'.format(cap[0],score))

    return score

for prompt in korean_prompt_list:
    #helsinki = helsinki model(prompt)
    #our_image = cross_attention_model(helsinki)
    #our_clip_score = calculate_clip_score(prompt,our_image)

    #bingsu_image = bingsu_model(prompt)
    #bingsu_clip_score = calculate_clip_score(prompt,bingsu_image)

    #print(our_clip_score, bingsu_clip_score)

korean_prompt = ['naive prompt','']
bingsu_image_path = ''

text = '기계번역 model 결과'
ko2eng = ['machine translation prompt',text]



