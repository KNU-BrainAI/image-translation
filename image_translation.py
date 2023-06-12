from transformers import MarianMTModel, MarianTokenizer
import torch
from cac_v1_5 import *

def translate_to_english(prompt):
    device = torch.device('cpu')  # Use CPU for translation
    model_name = 'Helsinki-NLP/opus-mt-ko-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    translated = model.generate(**inputs, max_new_tokens=512) 
    translated_prompt = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_prompt

#user_prompt = input("사용자의 프롬프트를 입력하세요: ")

#prompt = translate_to_english(user_prompt)
#print(prompt)


from keybert import KeyBERT
from kiwipiepy import Kiwi
from transformers import BertModel

def keyword_extraction(prompt):
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    kw_model = KeyBERT(model)
    keywords = kw_model.extract_keywords(prompt, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=len(prompt))
    p = [w[0] for w in keywords]
    return p

#prompts = keyword_extraction('The cat sitting on a rock')
#print(prompts)

#prompt = ['The cat sitting on a rock']
prompt = '돌 위에 앉아있는 고양이'

def generate_src_img(prompt):
  prompt = [prompt]
  #prompt = [translate_to_english(prompt)]
  g_cpu = torch.Generator().manual_seed(8888)
  controller = AttentionStore()
  image, x_t = run_and_display(prompt, controller, latent=None, run_baseline=False, generator=g_cpu)
  show_cross_attention(prompt, controller, res=16, from_where=("up", "down"))
  
  return x_t

x_t = generate_src_img(prompt)
#=================요청된 prompt 수정방법1===================================================================
#x_t = generate_src_img(prompt)

#prompts = ["The cat sitting on a rock",
#           "The dog sitting on a rock"]


#controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                             self_replace_steps=.4)
#_ = run_and_display(prompts, controller, latent=x_t)


#=================요청된 prompt 수정방법2===================================================================
#x_t = generate_src_img(prompt)

#prompts = ["The cat sitting on a rock",
#           "The dog sitting on a rock"]

#controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
#_ = run_and_display(prompts, controller, latent=x_t, run_baseline=True)

#=================요청된 prompt 수정방법3===================================================================
#x_t = generate_src_img(prompt)

#prompts = ["The cat sitting on a rock",
#           "The dog sitting on a rock"]
#lb = LocalBlend(prompts, ("cat", "dog"))
#controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS,
#                              cross_replace_steps={"default_": 1., "lasagne": .2},
#                              self_replace_steps=0.4,
#                              local_blend=lb)
#_ = run_and_display(prompts, controller, latent=x_t, run_baseline=True)

#=================요청된 prompt 수정방법4===================================================================
#prompts = ["The cat sitting on a rock", "The dog sitting on a rock"]


#equalizer = get_equalizer(prompts[1], ("dog",), (2,))
#controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                               self_replace_steps=.4,
#                               equalizer=equalizer)
#_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)

#======================================================함수로정의=============================================
def generate_trg_img(prompt,correct):
  
  prompt = translate_to_english(prompt).replace('.','')
  #correct = "cat->dog"
  command = correct.split('->')
  prev,curr = command[0], command[1]
  prev = translate_to_english(prev).replace('.','').lower()
  curr = translate_to_english(curr).replace('.','').lower()
  
  
  correct_prompt = prompt.replace(prev,curr)
  prompts = [prompt,correct_prompt]
  
  equalizer = get_equalizer(prompts[1],(curr,),(2,))
  controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=.4, equalizer=equalizer)
  
  _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)

correct = "고양이->개"
#generate_trg_img(prompt,correct)

def generate_test1(prompt,correct):
  
  prompt = translate_to_english(prompt).replace('.','')
  #correct = "cat->dog"
  command = correct.split('->')
  prev,curr = command[0], command[1]
  prev = translate_to_english(prev).replace('.','').lower()
  curr = translate_to_english(curr).replace('.','').lower()
  
  
  correct_prompt = prompt.replace(prev,curr)
  prompts = [prompt,correct_prompt]
  
  equalizer = get_equalizer(prompts[1],(curr,),(2,))
  controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=.4, equalizer=equalizer)
  
  _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


def generate_test(prompt,correct):
  
  prompt = translate_to_english(prompt).replace('.','')
  #correct = "cat->dog"
  command = correct.split('->')
  prev,curr = command[0], command[1]
  prev = translate_to_english(prev).replace('.','').lower()
  curr = translate_to_english(curr).replace('.','').lower()
  
  
  correct_prompt = prompt.replace(prev,curr)
  prompts = [prompt,correct_prompt]
  
  controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
  _ = run_and_display(prompts, controller, latent=x_t, run_baseline=True)
  
generate_trg_img(prompt,correct)