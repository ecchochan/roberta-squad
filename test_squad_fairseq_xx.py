from torch import nn
import argparse
import torch
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


max_seq_length   = 512
max_query_length = 128
doc_stride       = 128
merge_style      = 0

default_choices = []


import sys
eval_model = sys.argv[1]
eval_dir = sys.argv[2]

lang = sys.argv[3]
if lang == 'zh':
    default_choices = ['是','否']
elif lang == 'en':
    default_choices = ['yes','no']



from tokenization import FairSeqSPTokenizer, char_anchors_to_tok_pos
get_tokenizer = lambda: FairSeqSPTokenizer('xlmr.large.v0')

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

def read(dat):
    uid, inp, start, end, p_mask, unanswerable = marshal.loads(dat)
    inp = np.frombuffer(inp, dtype=np.uint16).astype(np.int32)
    p_mask = np.frombuffer(p_mask, dtype=np.bool).astype(np.float32)
    return uid, inp, start, end, p_mask, unanswerable

def fread(f):
    uid, inp, start, end, p_mask, unanswerable = marshal.load(f)
    inp = np.frombuffer(inp, dtype=np.uint16).astype(np.int32)
    p_mask = np.frombuffer(p_mask, dtype=np.bool).astype(np.float32)
    return uid, inp, start, end, p_mask, unanswerable
            
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
def work(ss, debug=False):
    
    unique_index, \
     context, \
     qas, \
     is_training, \
     return_feature = ss
    
    rss = tokenizer.merge_cq(context, 
                             qas,
                             max_seq_length = max_seq_length,
                             max_query_length = max_query_length,
                             doc_stride = doc_stride,
                             default_choices = default_choices,
                             unique_index=unique_index,
                             is_training=is_training,
                             merge_style=merge_style,
                             debug = debug
                           )
    o = 0
    results = []
    for rs in rss:
        q = qas[o]
        o += 1
        for r in rs:
            inp = tk.convert_tokens_to_ids(r.all_doc_tokens)
            start_position,end_position = char_anchors_to_tok_pos(r)
            p_mask = r.p_mask
            uid = r.unique_index[0]*1000 + r.unique_index[1]
            if start_position == -1 and end_position == -1:
                start_position = 0
                end_position = 0
                
            
            no_ans = start_position == 0
            
            #if no_ans:
            #    print(q['answer_text'], '>>', r.all_doc_tokens[start_position:end_position+1])
            assert start_position >= 0 and end_position >= 0 and start_position < len(inp) and end_position < len(inp)
            assert len(inp) <= max_seq_length
            record = marshal.dumps(
                (
                uid,
                np.array(inp,dtype=np.uint16).tobytes(),
                start_position,
                end_position,
                np.array(p_mask,dtype=np.bool).tobytes(),
                int(no_ans)
                )
            )
            
            if return_feature:
                results.append((record, no_ans, r.serialize()))
            else:
                results.append((record, no_ans))


    
    return results




def generate_tfrecord(data_dir,
                      write_fn=None, 
                      is_training=False,
                      return_feature=False,
                      parallel_process=False,
                      debug=False):
    global count

    if return_feature:
        rs = []

    i = 0
    
    if parallel_process:
        cpu_count = multiprocessing.cpu_count()
    
        pool = Pool(cpu_count-1,initializer=init)
        
    tokenizer = tk = get_tokenizer()
        
    tot_num_no_ans = 0
    
    
        
    records = []
    
        
    num_no_ans = 0
    i += 1

    jobs = ((i, c, q, is_training, return_feature) for i, c, q in gen(data_dir))
    t0 = time()
    results = pool.imap_unordered(work,jobs) if parallel_process else tqdm(iter(work(e, debug=debug) for e in jobs))
    c = 0
    for e in results:
        for record in e:
            if return_feature:
                record, no_ans, r = record
                r = tk.from_bytes(r)
                rs.append(r)
            else:
                record, no_ans = record


            records.append(record)

            if no_ans:
                num_no_ans += 1
            c += 1
            if c % 2500 == 0:
                t1 = time()
                uid, inp, start, end, p_mask, unanswerable = read(record)
                # print(uid, tk.convert_ids_to_tokens(inp) , start, end, p_mask)
                print('%d features (%d no ans) extracted (time: %.2f s)'%(c, num_no_ans, t1-t0))

    if not return_feature:
        random.shuffle(records)
        with open(write_fn, 'wb') as f:
            for record in records:
                f.write(record)
                f.write(b'\n')
    tot_num_no_ans = num_no_ans

    print('num has ans / num no ans : %d / %d'%(c - tot_num_no_ans, tot_num_no_ans))
    
    
    if return_feature:
        return records, rs
    


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


# Eval Utilities

import collections
_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob", "this_paragraph_text",
    "cur_null_score"])
_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob","cur_null_score"])

import math
def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


# Model Init


##############################################################################
##############################################################################
####
####   Below are using DataParallel... which is slow... 
####   and I do not know how to use DistributedDataParallel yet
####
##############################################################################
##############################################################################



from fairseq_train import RobertaQAModel
from time import time

roberta_single = RobertaQAModel.from_pretrained(roberta_directory, checkpoint_file=eval_model, strict=True).model




log_steps = 500
num_epochs = 2
max_seq_length = 512
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
  
  orig_data = {} 
  for e in gen(eval_dir):
    for q in e[2]:
      orig_data[q['id']] = q
  
  records, rs = generate_tfrecord(eval_dir, is_training=False, parallel_process=True, return_feature=True)

  records = records #[:100]
  rs = rs #[:100]


  batches = list(zip(from_records(records,batch_size, half=fp16, shuffle=False), chunks(rs,batch_size)))

  prediction_by_qid = {}
  
  with torch.no_grad():
    for e, rs in tqdm(batches):
      inp, p_mask, start, end, _ = e
      (start_logits, end_logits, cls_logits), _ = roberta(inp.to(device=device))
      start_logits = start_logits.squeeze(-1)
      end_logits = end_logits.squeeze(-1)

    
      for result, r in zip(zip(*(start_logits, end_logits, cls_logits)), rs):
        qid = r.qid
        if qid not in prediction_by_qid:
          prediction_by_qid[qid] = []
        prediction_by_qid[qid].append((result, r))

  
  return orig_data, prediction_by_qid
  

from squad_evaluation import compute_f1, normalize_answer

def handle_prediction_by_qid(self, 
                             prediction_by_qid, 
                             start_n_top = 5,
                             end_n_top = 5,
                             n_best_size = 5,
                             threshold = -1.5,
                             max_answer_length = 48,
                             debug = False,
                             wrong_only = False):
  global prelim_predictions
  use_ans_class = True
  all_predictions = {}
  all_predictions_output = {}
  scores_diff_json = {}
  score = 0
  for qid, predictions in tqdm(prediction_by_qid.items()):
    q = orig_data[qid]
    ri = 0
    prelim_predictions = []
    for result, r in predictions:
      paragraph_text = r.original_text
      original_s, original_e = r.original_text_span # exclusive
      this_paragraph_text = paragraph_text[original_s:original_e]
      cur_null_score = -1e6
      sub_prelim_predictions = []
      if use_ans_class:
        start_top_log_probs, end_top_log_probs, cls_logits = result
        cur_null_score = cls_logits.tolist()
      else:
        start_top_log_probs, end_top_log_probs = result
      if True:
        start_top_log_probs = start_top_log_probs.cpu().detach().numpy()
        end_top_log_probs = end_top_log_probs.cpu().detach().numpy()
        start_top_index = start_top_log_probs.argsort()[-start_n_top:][::-1].tolist()
        end_top_index = end_top_log_probs.argsort()[-end_n_top:][::-1].tolist()
        start_top_log_probs = start_top_log_probs.tolist()
        end_top_log_probs = end_top_log_probs.tolist()
        for start_index in start_top_index:
            for end_index in end_top_index:
              if start_index == 0 or end_index == 0:
                continue
              if end_index < start_index:
                continue
              if start_index >= len(r.segments) or end_index >= len(r.segments):
                continue
              seg_s = r.segments[start_index]
              seg_e = r.segments[end_index]
              if seg_s != seg_e:
                continue
              if r.is_max_context[start_index] == 0 :
                continue
              length = end_index - start_index + 1
              if length > max_answer_length:
                continue
              start_log_prob = start_top_log_probs[start_index]
              end_log_prob = end_top_log_probs[end_index]
              sub_prelim_predictions.append(
                  _PrelimPrediction(
                      feature_index=ri,
                      start_index=start_index,
                      end_index=end_index,
                      start_log_prob=start_log_prob,
                      end_log_prob=end_log_prob,
                      this_paragraph_text=this_paragraph_text,
                      cur_null_score=cur_null_score
                  ))
      prelim_predictions.extend(sub_prelim_predictions)
      ri += 1
    prelim_predictions = sorted(
        prelim_predictions,
        key=(lambda x: (x.start_log_prob + x.end_log_prob)),
        reverse=True)
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
          break
      r = predictions[pred.feature_index][1]
      cur_null_score = pred.cur_null_score
      this_paragraph_text = pred.this_paragraph_text
      s,e = pred.start_index, pred.end_index  # e is inclusive
      char_s  = r.tok_to_char_offset[s]
      char_e  = r.tok_to_char_offset[e]  # inclusive
      char_e += len(r.all_text_tokens[r.char_to_tok_offset[char_e]])
      final_text = r.text[char_s:char_e].strip() # this_paragraph_text[char_s:char_e]
      if False:
        print(final_text, '>>', r.all_text_tokens[s:e+1])
      if final_text in seen_predictions:
          continue
      seen_predictions[final_text] = True
      nbest.append(
        _NbestPrediction(
            text=final_text,
            start_log_prob=pred.start_log_prob,
            end_log_prob=pred.end_log_prob,
            cur_null_score=cur_null_score))
    if len(nbest) == 0:
        nbest.append(
          _NbestPrediction(text="", start_log_prob=-1e6,
          end_log_prob=-1e6,
          cur_null_score=-1e6))
    total_scores = []
    best_non_null_entry = None
    best_null_score = None
    best_score_no_ans = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry
        best_null_score = entry.cur_null_score
        best_score_no_ans = entry.cur_null_score
    probs = _compute_softmax(total_scores)
    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)
    s = compute_f1(normalize_answer(q['answer_text']), normalize_answer(best_non_null_entry.text) if best_null_score < threshold else '')
    all_predictions_output[qid] = [q['answer_text'], best_non_null_entry.text, best_null_score, s]
    if debug:
      ans = normalize_answer(best_non_null_entry.text) if best_null_score < threshold else '*No answer*'
      truth = normalize_answer(q['answer_text']) or '*No answer*'
      if (not wrong_only or ans != truth):
        print('Q:', q['question'])
        print('A:', ans, '(',best_null_score,')',  '[',best_score_no_ans,']', )
        print('Truth:', truth)
        print('')
      score += s
    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None
    all_predictions[qid] = best_non_null_entry.text
    scores_diff_json[qid] = best_null_score
  print('score: ', score, '/', len(all_predictions), '=', score / len(all_predictions))
  return nbest_json, all_predictions, scores_diff_json, all_predictions_output

  
try:
  orig_data, prediction_by_qid = evaluate(eval_dir)
  nbest_json, all_predictions, scores_diff_json, all_predictions_output = handle_prediction_by_qid(roberta_single, prediction_by_qid, threshold=-6.1, debug=False, wrong_only=True)
  
  with open('all_predictions_output.json','w') as f:
    json.dump(all_predictions_output,f, separators=(',',':'))
  
  from squad_evaluation import evaluate as squad_squad_evaluation
  with open(eval_dir, "r") as f:
    predict_data = json.load(f)["data"]
  result, exact_raw, f1_raw, wrongs = squad_squad_evaluation(predict_data, 
                                             all_predictions, 
                                             na_probs=scores_diff_json, 
                                             na_prob_thresh=0, 
                                             out_file=None, 
                                             out_image_dir=None)

finally:
  import code
  code.interact(local=locals())