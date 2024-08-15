# -*- coding: utf-8 -*-
"""
@Time: 02/07/2024 10:08
@Author: Yue
@Contact: Vancher0117@gmail.com
@Title: inference.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
from MyData import get_csv_data


# Define function to load data from '.csv' files with index
def load_question(df,idx):
   instruction,input,output = df.iloc[idx]['instruction'],df.iloc[idx]['input'],df.iloc[idx]['output']
   
   prompt = f"""
   结合中国文化背景和提示，回答问题。
   
   问题：{instruction}\n{input}
   
   答案：
   """

   print(f'INPUT PROMPT:\n{prompt}')
   print()
   print(f'BASELINE HUMAN ANSWER:\n{output}\n')

   return prompt


# Define model inference function
def model_inference(question,model,tokenizer):
   ipt = tokenizer(question,return_tensors='pt').to(model.device)
   opt = model.generate(**ipt, max_new_tokens=128, do_sample=True)   # To avoid repetitive, boring outputs
   answer = tokenizer.decode(opt[0], skip_special_tokens=True)
   print(f'MODEL GENERATION - ZERO SHOT:\n{answer}')

   # Free up GPU memory
   del ipt,opt,answer,
   torch.cuda.empty_cache()


# Define function to load Trained Model before inference
def load_mymodel(path,Base_LLM):
   model = PeftModel.from_pretrained(model=Base_LLM,model_id=path,is_trainable=False)
   return model




# Testing
if __name__ == '__main__':
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print('DEVICE: ',device)

   path = './result'
   MODEL_NAME = "m-a-p/CT-LLM-Base"
   print("Loading base model and its tokenizer...")
   model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME,trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False,trust_remote_code=True)
   if tokenizer.pad_token_id is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id

   #print('Loading trained model...')
   #model = load_mymodel(path,model_base)

   # Method 1:
   # Sample a question from '.csv' file
   #data_path = './Datasets/exam.csv'
   #df = get_csv_data(data_path)
   #prompt = load_question(df,1000)

   # Method 2:
   question = "问题：暗度陈仓的成语释义：\n答案："
   print(f'INPUT QUESTION:\n{question}')
   model_inference(question,model_base,tokenizer)