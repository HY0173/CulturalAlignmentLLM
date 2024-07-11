# -*- coding: utf-8 -*-
"""
@Time: 27/06/2024 10:31
@Author: Yue
@Contact: Vancher0117@gmail.com
@Title: MyData.py
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Define function to load data from '.csv' files
def get_csv_data(path):
  df = pd.read_csv(path)
  df = df.fillna('')
  df = df.drop(columns=['Unnamed: 0'])
  return df

IGNORE_INDEX = -100

# Create a custom Dataset
class MyDataset(Dataset):
  def __init__(self, data_csv, tokenizer):
    self.df = data_csv
    self.tokenizer = tokenizer

  def __getitem__(self, index):
    data = self.df.iloc[index]
    instruction,input,output = data['instruction'],data['input'],data['output']

    # For QA questions, input == ""
    if input is not None and input != "":
      instruction = instruction+'\n'+input

    # Regularization
    # Here, the terms '问题' and '答案' can be translated as 'Question' and 'Answer' respectively.
    source = f"问题：{instruction}\n答案："
    target = f"{output}{self.tokenizer.eos_token}"

    # Tokenize Source
    tokenzied_source = self.tokenizer(source,
                                      max_length=512,
                                      padding='max_length',
                                      truncation=True)
    # Tokenize Target
    tokenized_target = self.tokenizer(target,
                                      max_length=512,
                                      padding='max_length',
                                      truncation=True)
    
    # Get (input_ids, attention_mask, and labels) for Model Training
    ## torch.LongTensor(): int64 data type value
    ## [input_ids]
    input_ids = torch.LongTensor(tokenzied_source['input_ids'] + tokenized_target['input_ids'])

    ## [attention_mask]
    attention_mask = torch.LongTensor(tokenzied_source['attention_mask'] + tokenized_target['attention_mask'])

    ## [labels]
    labels = torch.LongTensor([IGNORE_INDEX] * len(tokenzied_source['input_ids']) + tokenized_target['input_ids'])

    return {
      "input_ids":input_ids,
      "attention_mask":attention_mask,
      "labels":labels
    }

  def __len__(self):
    return len(self.df)
  

# Testing 
if __name__ == 'main':
    data_path = './Datasets/exam.csv'
    TOKENIZER_NAME = 'm-a-p/CT-LLM-Base'

    df = get_csv_data(data_path)
    print('Data loaded.')

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME,use_fast=False,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
       tokenizer.pad_token_id = tokenizer.eos_token_id
    print('Tokenizer loaded.')

    dataset = MyDataset(df,tokenizer)
    print('Dataset loaded.\n')
    
    print('A sample in dataset: \n')
    print(dataset[0])