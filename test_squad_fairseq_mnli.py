from torch import nn
import argparse
import torch
from tokenizer.roberta import RobertaTokenizer, MASKED, NOT_MASKED, IS_MAX_CONTEXT, NOT_IS_MAX_CONTEXT, DocTokens
from glob import glob
import numpy as np
import json
from tokenizer.validate import validate
from copy import deepcopy
from time import time
from multiprocessing import Pool
import multiprocessing
import gc
import random
from tqdm import tqdm
import os

roberta_directory = './roberta.large'


max_query_length = 128
doc_stride       = 128
merge_style      = 0

default_choices = []
get_tokenizer = lambda: RobertaTokenizer(config_dir=roberta_directory)

tk = tokenizer =  get_tokenizer()


#Data Utilities


def init():
    global tokenizer, tk
    import gc
    tokenizer = tk = get_tokenizer()
    

def data_from_path(train_dir):
    index = 0
    for fn in glob(train_dir):
        with open(fn, "r") as f:
            entries = [e for e in json.load(f)["data"] for e in e['paragraphs']]


        print("%-40s : %s contexts"%(fn.split('/')[-1],len(entries)))
        for e in entries:
            c = e['context']
            yield index, c, e['qas']
            index += 1

def char_anchors_to_tok_pos(r):
    if len(r.char_anchors) == 2:
        a,b = r.char_anchors
    else:
        return -1,-1
    a = r.char_to_tok_offset[a]
    b = r.char_to_tok_offset[b]
    while b+1 < len(r.all_doc_tokens) and r.all_text_tokens[b+1] == '':
        b += 1
        
    return a, b

import marshal
def read(dat):
    inp, label = marshal.loads(dat)
    inp = np.frombuffer(inp, dtype=np.uint16).astype(np.int32)
    return inp, label

def fread(f):
    inp, label = marshal.load(f)
    inp = np.frombuffer(inp, dtype=np.uint16).astype(np.int32)
    return inp, label
            
def gen(paths):
    j = 0
    for i,context,qas in data_from_path(paths):
        for q in qas:
            if len(q['question']) < 5 or ('choices' in q and ''.join(q['choices']) == ''):
                continue
            if '\1' in q['question']:
                q['question'] = q['question'].replace('\1', '___')
        #j += len(qas)
        #if j > 1000:
        #  return
        yield i,context, qas
        
        
        
import marshal


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

def pad(list_of_tokens, 
        dtype=np.long,
        torch_tensor=None,
        pad_idx=1):
    k = np.empty((len(list_of_tokens),max_seq_length), dtype=dtype)
    k.fill(pad_idx)
    i = 0
    for tokens in list_of_tokens:
        k[i,:len(tokens)] = tokens
        i += 1
    return k if torch_tensor is None else torch_tensor(k)

  
def from_records(records, batch_size = 48, half=False, shuffle=True):
    if half:
      float = torch.HalfTensor
    else:
      float = torch.FloatTensor
  
  
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

    if shuffle:
      records = list(records)
      random.shuffle(records)
    for record_samples in chunks(records,batch_size):
        uid, inp, start, end, p_mask, unanswerable = zip(*record_samples) if fn_style else zip(*(read(record) for record in record_samples))
        start = torch.LongTensor(start)
        end = torch.LongTensor(end)
        unanswerable = float(unanswerable)
        inp = pad(inp,dtype=np.long, torch_tensor=torch.LongTensor)
        p_mask = pad(p_mask,dtype=np.float32, torch_tensor=float)

        yield inp, p_mask, start, end, unanswerable

# Train Utilities


# Model Init


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


from fairseq_train_mnli import RobertaMNLIModel
from time import time
roberta_directory = './roberta.large'
roberta_single = RobertaMNLIModel.from_pretrained(roberta_directory, checkpoint_file=eval_model+'.pt', strict=True).model




log_steps = 500
num_epochs = 2
max_seq_length = 256
num_cores = torch.cuda.device_count() # 8
effective_batch_size = 64             # 8  bs per device
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
  

  batches = from_records(eval_dir,batch_size, half=fp16, shuffle=False)

  count = 0

  ncorrects = 0
  
  with torch.no_grad():
    for inp, labels in tqdm(batches):
      cls_logits, _ = roberta(inp.to(device=device))

      preds = cls_logits.argmax(1).tolist()

      ncorrect = sum(a == b for a, b in zip(labels, preds))

      ncorrects += ncorrect
    
      count += len(inp)

  
  print('Accuracy: ', '%.4f'%(100*ncorrects / count), '%', '(',ncorrects,'/',count,')' )
  
  

evaluate(eval_dir)