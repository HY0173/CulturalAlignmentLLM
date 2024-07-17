# -*- coding: utf-8 -*-
"""
@Time: 13/07/2024 10:37
@Author: Yue
@Contact: Vancher0117@gmail.com
@Title: Train.py
"""

import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PeftModel
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from MyData import get_csv_data, MyDataset

# 1. Make path to save the model
out_path = './result'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# 2. Define function for Model Tuning
def cultural_train(args):
    # ========================================
    #           2.1 Data Preparation
    # ========================================
    print("Loading Data...")
    df = get_csv_data(args.data_path)                   # Load data from '.csv' file
    print("DATA LOADED.")

    # Split Data for Training (8:1:1)
    print("Data Splitting...")
    train_df,test_df = train_test_split(df,test_size=0.2,shuffle=True)
    print("Size of Validation data: ",len(train_df))
    val_df,test_df = train_test_split(test_df,test_size=0.5,shuffle=True)
    print("Size of Validation data: ",len(val_df))
    print("Size of Testing data: ",len(test_df))

    # Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME,use_fast=False,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("TOKENIZER LOADED.")

    # Convert Text to Tensor in Dataset format & Wrapped by DataLoader
    print("Text to Tensor...")
    train_dataset,val_dataset,test_dataset = MyDataset(train_df,tokenizer),MyDataset(val_df,tokenizer),MyDataset(test_df,tokenizer)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset,shuffle=True,batch_size=args.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=args.BATCH_SIZE)
    print("DATALOADER PREPARED.")


    # ========================================
    #            2.2 Model Setup
    # ========================================
    # (1) Initialize Base LLM
    print("Loading Base LLM...")
    model = AutoModelForCausalLM.from_pretrained(args.MODEL_NAME,trust_remote_code=True)
    print("BASE MODEL LOADED.")

    # (2) Hard/Soft Prompt config
    # Initialize the soft prompt with the following text:
    prompt_tuning_init_text = '结合中国文化背景和提示，回答问题。'
    peft_config = PromptTuningConfig(
        task_type = TaskType.CAUSAL_LM,
        prompt_tuning_init = PromptTuningInit.TEXT,
        prompt_tuning_init_text = '结合中国文化背景和提示，回答问题。',
        num_virtual_tokens = len(tokenizer(prompt_tuning_init_text)['input_ids']),
        tokenizer_name_or_path = args.MODEL_NAME,
    )

    print("Creating a PEFT MODEL...")
    model = get_peft_model(model,peft_config)
    print(model.print_trainable_parameters())
    

    # ========================================
    #            2.3 Prompt Tuning
    # ========================================
    print("Start Prompt Tuning "+args.LLM+" for Chinese Cultural Alignment.\n")
    # Move the model to GPU
    model = model.to(args.device)

    # Setup Optimizer & learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.epochs),
        )
    
    # Training Loop
    train_epoch_loss,eval_epoch_loss,train_epoch_ppl,eval_epoch_ppl =[],[],[],[]
    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch+1}")
        # ========================================
        #             Training Phase
        # ========================================
        model.train()
        train_loss = 0
        for step,batch in enumerate(tqdm(train_dataloader)):
            batch = {k:v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)

            # Calculate Loss
            loss = outputs.loss
            train_loss += loss.detach().float()
            
            # Gradient Calculation
            loss.backward()

            # Update
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Free up GPU Memory
            del batch,loss,
            torch.cuda.empty_cache()

        # Record Loss
        train_epoch_loss.append(train_loss/len(train_dataloader))
        print(f"Training Loss: {train_epoch_loss[-1]:.4f}")
        # Perplexity (the exponentiated average negative log-likelihood of a sequence)
        train_epoch_ppl.append(torch.exp(train_epoch_loss[-1]))
        print(f"Training PPL: {train_epoch_ppl[-1]:.4f}")

        # ========================================
        #            Validation Phase
        # ========================================
        print('Evaluate on Validation Data...')
        model.eval()
        eval_loss = 0
        for step,batch in enumerate(tqdm(val_dataloader)):
            batch = {k:v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
        
            # Free up GPU Memory
            del batch,loss,
            torch.cuda.empty_cache()
        
        eval_epoch_loss.append(eval_loss/len(val_dataloader))
        print(f"Validation Loss: {eval_epoch_loss[epoch]:.4f}")
        eval_epoch_ppl.append(torch.exp(eval_epoch_loss[-1]))
        print(f"Validation PPL: {eval_epoch_ppl[-1]:.4f}")

    # Save the model

    print("Done!")



if __name__ == 'main':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ',device)

    # Define Hyperparameters
    parser = argparse.ArgumentParser("")
    parser.add_argument("--out_path", default="./result")
    parser.add_argument("--data_path", default="./Datasets/exam.csv")
    parser.add_argument("--LLM", default="CT-LLM-Base")
    parser.add_argument("--MODEL_NAME", default="m-a-p/CT-LLM-Base")
    parser.add_argument("--device", default=device)
    parser.add_argument("--BATCH_SIZE", default=8, type=int)
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--epochs", default=5, type=int)

    args = parser.parse_args()

    # Start training
    cultural_train(args)
