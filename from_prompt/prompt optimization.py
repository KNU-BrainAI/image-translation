from transformers import MarianMTModel, MarianTokenizer
import gc
import torch 
  
gc.collect()
torch.cuda.empty_cache()

# In[2]:


#!pip install transformers 
#!pip install accelerate

# In[3]:

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# In[4]:


few_src = [] #학습시킬때 source 문장 입력. 여기서는 naive 문장
few_tgt = [] #학습시킬때 target 문장 입력. 여기서는 목표하는 style의 문장이나 단어 입력(원하는 template)


def put_datapair(src, tgt):
  few_src.append(src)
  few_tgt.append(tgt)


# k=4 일때 아래처럼 입력하면 됨
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


# k=4 일때 학습하는 pair 문장 예시
put_datapair("Hello. This is my first request, so it may be difficult. It's soon New Year's Day and the background is a full moon. And... and there's a black rabbit wearing a traditional Korean hanbok standing there. Could you please draw this for me as soon as possible?","The background is a full moon, and please draw a black rabbit wearing a traditional Korean hanbok, standing on New Year's Day.")
put_datapair("Long time no see. I've had a good idea. The background is a night sky with stars. Oh... what's it called? Oh right. Could you please draw a black rabbit making a wish while looking at a full moon? Thank you.","The background is a night sky with stars, and please draw a black rabbit making a wish while looking at a full moon.")
put_datapair("Nice to meet you. Is it possible to do this? The background is a castle with snow falling. Oh, I almost forgot. The black rabbit is a prince, and the white rabbit is a princess, and they are facing each other. Is it possible? Thank you.","a picture of a white snow-covered castle in the background, with a black rabbit as the prince facing a white rabbit as the princess, with the two of them looking at each other.")
put_datapair("Hi~. I'm sorry for the sudden request. The background is a galaxy and, um... And... Could you please draw a black rabbit riding on a sleigh. Please as soon as possible~.","The background is a galaxy, and please draw a black rabbit riding on a sleigh.")

# 똑바로 입력했는지 확인차 출력
print(few_src)
print(few_tgt)


# ### In-Context Learning
# 
# ***Reference: [A Recipe for Arbitrary Text Style Transfer with Large Language Models, (Reif et al. (Google Research), ACL 2022)](https://aclanthology.org/2022.acl-short.94/)***
# 탑티어 학회 논문에서 발췌하여 같은 방식의 문장으로 모델을 훈련시킴.
# In[5]:

# NLP 모델이 이해하기 쉽도록 문장을 지정. 이것도 자유롭게 변경할 수 있음.
in_context_text = ''

for i in range(len(few_src)):
  in_context_text += f'Here is some text: {few_src[i]}. Here is a rewrite of the text, which is more simple: {few_tgt[i]}\n'

print(in_context_text)

# transformers model 불러오기

# In[6]:


from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import pipeline


# GPT-2 (This is the smallest version of GPT-2, with 124M parameters.) = "gpt2"
# GPT-2 (XL) (1.5B parameters.) = "gpt2-xl"
# T5 (base, with 220 million parameters.) = "t5-base"
# T5 (3B parameters.) = "t5-3b"
# T5 (11B parameters.) = "t5-11b" -> 45.2GB (Colab may not cover this size..)
# GPT-J (6B parameters.) = "EleutherAI/gpt-j-6B"

# In-context learning work well at least 1 Billions parameters. -eunchan-

model_zoo = ["gpt2","gpt2-xl","t5-base","t5-3b","t5-11b","EleutherAI/gpt-j-6B"]

model_name = model_zoo[5] #gpt-j

#T5 모델 같은 경우는 gpt 기반 모델과 다르기 때문에 이렇게 처리
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
def translate_to_english(prompt):
    device = torch.device('cpu')  # Use CPU for translation
    model_name = 'Helsinki-NLP/opus-mt-ko-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    translated = model.generate(**inputs, max_new_tokens=512) 
    translated_prompt = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_prompt

user_prompt = input("사용자의 프롬프트를 입력하세요: ")
english_prompt = translate_to_english(user_prompt)

#test_input_text = 'kjkjkjkj'
test_input_text = english_prompt
test_output_length = 16 #token length 자유롭게 지정



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
