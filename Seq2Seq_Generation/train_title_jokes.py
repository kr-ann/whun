import datetime
import os
import time
import sys

import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import Trainer, TrainingArguments
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

"""# Load model checkpoint from huggingface Library

Load the model which you want to use and load the tokenizer for that model.

"""

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

"""# Download and Prepare Data

Download the data from github repo. Load the dataset from the .json file and remove the unwanted columns. Divide the dataset for training and validation. Use the categories in validation data to generate jokes.
"""

#!git clone https://github.com/taivop/joke-dataset.git

y = pd.read_json('joke-dataset/wocka.json')
del y['id']
del y['category']

z = pd.read_json('joke-dataset/stupidstuff.json')
del z['id']
del z['score']

sum_data = pd.concat([y,z])
sum_data

sum_data = sum_data.sample(len(sum_data), random_state=20)
train_sub = int(len(sum_data) * 0.99)

train_df = sum_data[0:train_sub]
val_df = sum_data[train_sub:]

train_texts = list(train_df['title'])
val_texts = list(val_df['title'])

train_decode = list(train_df['body'])
val_decode = list(val_df['body'])

"""# Tokenize

Tokenize the data and convert them to a pytorch data object for training.
"""

train_encodings = tokenizer(train_texts, max_length=64, truncation=True, padding='longest')
val_encodings = tokenizer(val_texts, max_length=64, truncation=True, padding='longest')

train_labels = tokenizer(train_decode, max_length=256, truncation=True, padding='longest')
val_labels = tokenizer(val_decode, max_length=256, truncation=True, padding='longest')

class Summary_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

train_dataset = Summary_dataset(train_encodings, train_labels)
val_dataset = Summary_dataset(val_encodings, val_labels)

"""# Training"""

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=100,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    logging_steps=5,
    eval_accumulation_steps=1,
    learning_rate=1e-4,
    adafactor = True                #use adafactor instead of adam
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

trainer.save_model('pegasus_jokes')