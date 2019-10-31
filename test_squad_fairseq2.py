from torch import nn
import argparse
import torch
from tokenizer.roberta import RobertaTokenizer, MASKED, NOT_MASKED, IS_MAX_CONTEXT, NOT_IS_MAX_CONTEXT, DocTokens
from glob import glob
import numpy as np
import json
from copy import deepcopy
from time import time
from multiprocessing import Pool
import multiprocessing
import gc
import random
from tqdm import tqdm
import os

# specially made for roberta
from tokenizer.roberta import RobertaTokenizer, MASKED, NOT_MASKED, IS_MAX_CONTEXT, NOT_IS_MAX_CONTEXT
from tokenizer.validate import validate


roberta_directory = './roberta.large'


max_seq_length   = 192
max_query_length = 128
doc_stride       = 128

default_choices = []
get_tokenizer = lambda: RobertaTokenizer(config_dir=roberta_directory)

tk = tokenizer =  get_tokenizer()


#Data Utilities

import marshal
def read(dat):
    a, b = marshal.loads(dat)
    a = np.frombuffer(a, dtype=np.uint16).astype(np.int32)
    b = np.frombuffer(b, dtype=np.uint16).astype(np.int32)
    return a, b

def fread(f):
    a, b = marshal.load(f)
    #a = np.frombuffer(a, dtype=np.uint16).astype(np.int32)
    #b = np.frombuffer(b, dtype=np.uint16).astype(np.int32)
    return a, b
            

def pad(list_of_tokens, 
        dtype=np.long,
        torch_tensor=None,
        max_seq_length=max_seq_length,
        pad_idx=1):
    k = np.empty((len(list_of_tokens),max_seq_length), dtype=dtype)
    k.fill(pad_idx)
    i = 0
    for tokens in list_of_tokens:
        k[i,:len(tokens)] = tokens
        i += 1
    return k if torch_tensor is None else torch_tensor(k)

from torch.utils.data.dataset import Dataset


def chunks(l, n):
    if type(l) == type((e for e in range(1))):
        it = iter(l)
        while True:
            out = []
            try:
                for _ in range(n):
                    out.append(next(it))
            except StopIteration:
                yield out
                break

            yield out
    else:
    
        for i in range(0, len(l), n):
            yield l[i:i + n]

def from_records(records):
    """
    Args:
        records (string): Path to the csv file with annotations.
    """
  
    fn_style = isinstance(records,str)
    if fn_style:
      def from_file(fn):
        with open(fn, 'rb') as f:
            while True:
                try:
                    record = fread(f)
                    yield record
                except EOFError:
                    break
      records = from_file(records)

    records = list(records)
      
    prepared_records = []
    for record_samples in chunks(records,48):
        a, b = zip(*record_samples) if fn_style else zip(*(read(record) for record in record_samples))
        #a = pad(a,dtype=np.long, torch_tensor=torch.LongTensor)
        #b = pad(b,dtype=np.long, torch_tensor=torch.LongTensor)

        for e in zip(a,b):
            yield e











##############################################################################
##############################################################################
####
####   Below are using DataParallel... which is slow... 
####   and I do not know how to use DistributedDataParallel yet
####
##############################################################################
##############################################################################



import sys
eval_model = sys.argv[1]
eval_dir = sys.argv[2]


from fairseq_train2 import RobertaQAEmbedModel
from time import time
roberta_directory = './roberta.large'
roberta_single = RobertaQAEmbedModel.from_pretrained(roberta_directory, checkpoint_file=eval_model+'.pt', strict=True).model




log_steps = 500
num_epochs = 2
max_seq_length = 512
num_cores = torch.cuda.device_count() # 8
effective_batch_size = 512             # 8  bs per device
update_freq = 1                       # 4  bs per device
fp16 = True
class args:
  update_freq=update_freq
  fp16_scale_window=128
  distributed_world_size=1
  fp16_init_scale=4
  fp16_scale_tolerance=0
  threshold_loss_scale=1
  min_loss_scale=1e-4
  
  

use_gpu = None

assert effective_batch_size % update_freq == 0

batch_size = effective_batch_size // update_freq



if num_cores > 1:
  roberta = nn.DataParallel(roberta_single)

  
print("Let's use", num_cores, "GPUs!")

use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu

device = torch.device("cuda:0" if use_gpu else "cpu")


if not use_gpu:
  fp16 = False


roberta.to(device)

if fp16:
  roberta.half()
  
roberta.eval()
  
  
  
  
  
def evaluate(eval_dir):
  

  questions = []
  answers = []


  for q, a in tqdm(from_records(eval_dir)):
    questions.append(q)
    answers.append(a)
    

  batches = zip(chunks(questions,batch_size), chunks(answers,batch_size))
  correct_count = 0
  total_count = 0

  with torch.no_grad():
    with tqdm(batches) as t:
      for qs, ans in t:
        q = [np.frombuffer(e, dtype=np.uint16).astype(np.int32) for e in qs]
        a = [np.frombuffer(e, dtype=np.uint16).astype(np.int32) for e in ans]
        
        q = pad(q,dtype=np.long, torch_tensor=torch.LongTensor, max_seq_length=max(len(e) for e in q)).cuda()
        a = pad(a,dtype=np.long, torch_tensor=torch.LongTensor, max_seq_length=max(len(e) for e in a)).cuda()
        
        (loss, corrects) = roberta(q, a, return_loss=True)
        correct_count += corrects.sum().tolist()
        total_count   += len(q)
        t.set_description('accuracy: %.6f'%(correct_count/total_count))
      
  print('accuracy: %.6f'%(correct_count/total_count))
  

evaluate(eval_dir)