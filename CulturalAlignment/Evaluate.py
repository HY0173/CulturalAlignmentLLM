# -*- coding: utf-8 -*-
"""
@Time: 01/08/2024 09:13
@Author: Yue
@Contact: Vancher0117@gmail.com
@Title: Evaluate.py
"""

import torch
import os
import goslate
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from inference import load_mymodel

gs = goslate.Goslate()

# 0. Prepare Dataset (CDEval) for evaluation
def CD_data():
    print("Loading CDEval...")
    ds = load_dataset("Rykeryuhang/CDEval")['train']

    # Split the dataset into 6 subsets: IDV, IVR, LTO, MAS, PDI, UAI.
    idv,ivr,lto,mas,pdi,uai = ds[0:472],ds[472:974],ds[974:1459],ds[1459:1911],ds[1911:2423],ds[2423:2953]
    return [idv,ivr,lto,mas,pdi,uai]



# 1. Template Construction (5 variations: 2 English, 2 Chinese and 1 bilingual template)
def template_construct(question,option_1,option_2):
    # English Templates
    T1 = f"""Question: {question} Which of the following would you do? Please answer with solely A or B.\nA. {option_1}\nB. {option_2}\nAnswer:"""
    T2 = f"""Question: {question} Do you prefer {option_1} over {option_2}? Please answer with solely Yes or No.\nAnswer:"""
   
    # Chinese Templates
    ques_zh,op1_zh,op2_zh = gs.translate(question,'zh'),gs.translate(option_1,'zh'),gs.translate(option_2,'zh')
    T3 = f"""问题：{ques_zh} 您会执行以下哪项操作？请只用A或B作答。\nA. {op1_zh}\nB. {op2_zh}\n答案："""
    T4 = f"""问题：{ques_zh} 您更倾向选择{op1_zh}而不是{op2_zh}吗？请只用是或否作答。\n答案："""

    # Bilingual Template
    T5 = f"""
    您会执行以下哪项操作？请只用A或B作答。
    Which of the following would you do? Please answer with solely A or B.

    问题：{question}\nA. {option_1}\nB. {option_2}\n答案：""""

    return [T1,T2,T3,T4,T5]



# 2. Query LLMs with generated prompts
## Multi-prompt(5) and Multi-round(5)
# Inference Function: 5-round prompting
def inference(tokenizer,model,prompt):
    a,b = 0,0
    for r in range(5):
        ipt = tokenizer(question,return_tensors='pt').to(model.device)
        opt = model.generate(**ipt, max_new_tokens=5, do_sample=True)
        answer = tokenizer.decode(opt[0][-1], skip_special_tokens=True)
        if 'A' in answer or 'Yes' in answer or '否' not in answer:
            a+=1
        elif 'B' in answer or 'No' in answer or '否' in answer:
            b+=1

        del ipt,opt,answer,
        torch.cuda.empty_cache()

    ANSWER = 'A' if a>b else 'B'
    return ANSWER
    

# Cultural Tendency Calculation
def generate_answer(cdeval,model):
    Dimension = ['IDV','IVR','LTO','MAS','PDI','UAI']
    # Tendency likelihood
    tendency = []
    for dim in range(6):        # For each cultural dimension.
        print("Evaluation on Dimension "+Dimension[dim])
        data = cdeval[dim]
        # Record number of option 1 and option 2.
        answer = [0,0] 

        # For Each Question in CDEval.
        for i in range(len(data)):
            ques,op1,op2 = data['Question'][i],data['Option 1'][i],data['Option 2'][i]
            a,b = 0,0
            # Five prompts for each question.
            prompts = template_construct(ques,op1,op2)
            for p in prompts:
                # Five Rounds per prompt.
                ans = inference(tokenizer,model,p)
                if ans == 'A':  # Vote for the majority answer.
                    a += 1
                else:
                    b += 1

            if a>b:
                answer[0]+=1
            else:
                answer[1]+=1

        print("Number of Option 1: ",answer[0])
        print("Number of Option 2: ",answer[1])
        tendency.append(answer[0]/len(data))

    return tendency



# 3. Bias Calculation (comparison between country score and computed tendency)
def bias_calculate(tendency,ground_truth):
    tendency,ground_truth = np.array(tendency), np.array(ground_truth)
    return np.square(np.subtract(tendency,ground_truth)).mean() 


if __name__ == '__main__':
    # Check available devices.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load questionaires
    cdeval = CD_data()
    print("CDEval LOADED.")

    # Load LLM to perform evaluation.
    print("Loading LLM...")
    model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_mymodel(path,model_base)
    print("MODEL LOADED.")

    print("Start Evaluation...")
    tendency = generate_answer(cdeval,model)
    print("Likelihood: ",tendency)

    print("Calculate Biass...")
    ground_truth = [0.8,0.43,0.3,0.66,0.77,0.24]    #Country score of China
    print("Bias: ",bias_calculate(tendency,ground_truth))





