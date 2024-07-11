# -*- coding: utf-8 -*-
"""
@Time: 04/07/2024 08:05
@Author: Yue
@Contact: Vancher0117@gmail.com
@Title: Train.py
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PeftModel
from sklearn.model_selection import train_test_split

from MyData import get_csv_data, MyDataset

# 1. Make path to save the model
out_path = './result'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# 2. Model Tuning
def cultural_train(args):
    # Load data from '.csv' file
    df = get_csv_data(args.data_path)
    print("DATA LOADED.")

    # Prepare Data for Training (8:1:1)
    print("Data Splitting...")
    train_df,test_df = train_test_split(df,test_size=0.2,shuffle=True)
    print("Size of Validation data: ",len(train_df))
    val_df,test_df = train_test_split(test_df,test_size=0.5,shuffle=True)
    print("Size of Validation data: ",len(val_df))
    print("Size of Testing data: ",len(test_df))

    # Load the Base LLM
    print("Prompt Tuning "+args.LLM+" for Chinese Cultural Alignment.\n")

    model = AutoModelForCausalLM.from_pretrained(args.MODEL_NAME,trust_remote_code=True)
    print("MODEL LOADED.")
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME,use_fast=False,trust_remote_code=True)
    print("TOKENIZER LOADED.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert Text to Tensor in Dataset format
    train_data = MyDataset(train_df,tokenizer)
    val_data = MyDataset(val_df,tokenizer)
    test_data = MyDataset(test_df,tokenizer)
    print("DATASET PREPARED.")

    # Define configs
    # (1) Hard/Soft Prompt config
    # Initialize the soft prompt with the following text:
    prompt_tuning_init_text = '结合中国文化背景和提示，回答问题。'

    peft_config = PromptTuningConfig(
        task_type = TaskType.CAUSAL_LM,
        prompt_tuning_init = PromptTuningInit.TEXT,
        prompt_tuning_init_text = '结合中国文化背景和提示，回答问题。',
        num_virtual_tokens = len(tokenizer(prompt_tuning_init_text)['input_ids']),
        tokenizer_name_or_path = args.MODEL_NAME,
    )

    # (2) Training config
    train_args = TrainingArguments(
        seed = 42,
        output_dir = args.out_path,
        per_device_train_batch_size = args.BATCH_SIZE,    # BATCH_SIZE
        per_device_eval_batch_size = args.BATCH_SIZE,
        #gradient_accumulation_steps = 8,
        logging_steps = 10,
        learning_rate = args.lr,                          # Learning_rate
        num_train_epochs = args.epochs,                   # Epoch
        )
    
    # Training
    # (1) Initialize model for prompt tuning.
    model = get_peft_model(model,peft_config)
    print(model.print_trainable_parameters())

    # Move model to GPU
    model = model.to(args.device)

    # (2) Define Trainer
    trainer = Trainer(
        model = model,                    # Model
        args = train_args,                # Training Configs
        train_dataset = train_data,       # Datasets
        eval_dataset = val_data,
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True),
        load_best_model_at_end = True,    # Save the best model after training
        save_strategy = 'steps',
    )

    # (3) Start Training
    print("Start Training...\n")
    trainer.train()

    print("Completed!")
        


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
