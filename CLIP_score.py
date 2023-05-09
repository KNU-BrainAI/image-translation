# %%
import torch
import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', device)

raw = ['Raw_prompt', 'The red car in the winter forest.high detailed, insane quality, 4k']
blip = ['BLIP_caption', 'a red car driving through a snowy forest']
any = ['Any_caption', 'a baby in the house']

caption_list = [raw,blip,any]


for cap in caption_list:

    image = preprocess(Image.open('./redcar.jpg')).unsqueeze(0).to(device)
    caption = clip.tokenize([cap[1]]).to(device)


    with torch.no_grad():
        image_embedding = model.encode_image(image).float()
        caption_embedding = model.encode_text(caption).float()


    score = 100 * float(torch.cosine_similarity(image_embedding, caption_embedding))

    print('{}과의 CLIP score : {}'.format(cap[0],score))
# %%=======================usign matmul==================================
import torch
import clip

# CLIP 모델과 preprocessor 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', device)

image = Image.open('./redcar.jpg')
caption = 'a red car driving through a snowy forest'
#aption = ' a baby in the house'
#caption = 'The red car in the winter forest.high detailed, insane quality, 4k'
# 이미지와 캡션 입력
image = preprocess(image).unsqueeze(0).to(device)
#caption = preprocess(caption).unsqueeze(0).to(device)
caption = clip.tokenize([caption]).to(device)

with torch.no_grad():
    image_embedding = model.encode_image(image)
    caption_embedding = model.encode_text(caption)

# 이미지와 캡션 임베딩 간의 유사도 계산
score = torch.matmul(image_embedding, caption_embedding.T)
print(score)
# %%
import torch
from torchmetrics.functional.multimodal import clip_score
from PIL import Image
import numpy as np

image = './redcar.jpg'

raw_prompt = 'The red car in the winter forest.high detailed, insane quality, 4k'
blip_caption = 'a red car driving through a snowy forest'
any_caption = ' a baby in the house'

def calculate_clip_score(image_path, caption):
    image = Image.open(image_path)
    img = torch.permute(torch.tensor(np.array(image)), (2,0,1))

    score = clip_score(img, caption, "openai/clip-vit-base-patch32")

    return score.detach()

calculate_clip_score(image, raw_prompt)
# %%
import gc
import torch 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

gc.collect()
torch.cuda.empty_cache()



few_src = [] 
few_tgt = []


def put_datapair(src, tgt):
  few_src.append(src)
  few_tgt.append(tgt)


# k=4
'''
put_datapair("k", "x")
put_datapair("kk", "xx")
put_datapair("kkk", "xxx")
put_datapair("kkkk", "xxxx")
'''


# k=4 -> Manual sample: Text style transfer
'''put_datapair("South Korea's Capital is Seoul", "Seoul -> South Korea")
put_datapair("Mexico's Capital is Mexico City", "Mexico City -> Mexico")
put_datapair("Germany's Capital is Berlin", "Berlin -> Germany")
put_datapair("Uganda's Capital is Kampala", "Kampala -> Uganda")'''


# k=...
put_datapair("Hello. This is my first request, so it may be difficult. It's soon New Year's Day and the background is a full moon. And... and there's a black rabbit wearing a traditional Korean hanbok standing there. Could you please draw this for me as soon as possible?","The background is a full moon, and please draw a black rabbit wearing a traditional Korean hanbok, standing on New Year's Day.")
put_datapair("Long time no see. I've had a good idea. The background is a night sky with stars. Oh... what's it called? Oh right. Could you please draw a black rabbit making a wish while looking at a full moon? Thank you.","The background is a night sky with stars, and please draw a black rabbit making a wish while looking at a full moon.")
put_datapair("Nice to meet you. Is it possible to do this? The background is a castle with snow falling. Oh, I almost forgot. The black rabbit is a prince, and the white rabbit is a princess, and they are facing each other. Is it possible? Thank you.","a picture of a white snow-covered castle in the background, with a black rabbit as the prince facing a white rabbit as the princess, with the two of them looking at each other.")
put_datapair("Hi~. I'm sorry for the sudden request. The background is a galaxy and, um... And... Could you please draw a black rabbit riding on a sleigh. Please as soon as possible~.","The background is a galaxy, and please draw a black rabbit riding on a sleigh.")



print(few_src)
print(few_tgt)

in_context_text = ''

for i in range(len(few_src)):
  in_context_text += f'Here is some text: {few_src[i]}. Here is a rewrite of the text, which is more simple: {few_tgt[i]}\n'

print(in_context_text)

#%%
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import pipeline


model_zoo = ["gpt2","gpt2-xl","t5-base","t5-3b","t5-11b","EleutherAI/gpt-j-6B"]



model_name = model_zoo[5]


if model_name.find('t5') > -1: #model = t5
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  nlg_pipeline = pipeline('text2text-generation',model=model, tokenizer=tokenizer)

else: #model = gpt
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)
  #nlg_pipeline = pipeline(model=model, tokenizer=tokenizer)

# model inference

# In[7]:



#test_input_text = 'kjkjkjkj'
test_input_text = "Hello, please draw an illustration. Um... and... a pink-haired girl wearing a hoodie with cat ears, around the age of 16, in the style of Japanese animation. Please."
test_output_length = 16 #token length



if model_name.find('t5') > -1: #model = T5
  def generate_text(pipe, text, num_return_sequences=5, max_length=512):
    text = f"{text}"
    out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length, num_beams=5, no_repeat_ngram_size=2,)
    return [x['generated_text'] for x in out]

  #target_text = 'kjkjkjkj'
  src_text = in_context_text + f"Here is some text: {test_input_text}\n. Here is a rewrite of the text, which is more simple: "

  print("Input text:", src_text)
  test_output_text = generate_text(nlg_pipeline, src_text, num_return_sequences=1, max_length=test_output_length)


  #you can cook this output anything you want!  
  print(test_output_text)



else:  #model = GPT
  src_text = in_context_text + f"Here is some text: {test_input_text}\n. Here is a rewrite of the text, which is more simple: "
  tokens = tokenizer.encode(src_text, return_tensors='pt')
  gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=len(tokens[0])+test_output_length)
  generated = tokenizer.batch_decode(gen_tokens)[0]
  
  test_output_text = generated[generated.rfind('more simple:')+12:]
  print(generated)

  #you can cook this output anything you want!
  print(test_output_text)