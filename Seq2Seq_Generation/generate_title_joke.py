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
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig

"""# Load saved model"""
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
config = PegasusConfig.from_json_file('./content/saved_model/*.config') #Path of .config file
model = PegasusForConditionalGeneration.from_pretrained('./content/saved_model/pytorch_model.bin', config=config).to(torch_device) #path of .bin file

"""# Generate Text

Generate text using different sets of arguments. You can find more on generating text here: 
[How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
"""

batch = tokenizer('If a train is traveling at 80 mph from Chicago to Cleveland, how may flapjacks does it take to cover the roof of a doghouse?', truncation=True, padding='longest', return_tensors="pt").to(torch_device)
generated = model.generate(**batch, min_length=64, do_sample=True, top_p=0.92, top_k=50, num_beams=8, no_repeat_ngram_size=1)
tgt_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

print(tgt_text)