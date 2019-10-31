# roBERTa training for SQuAD 

**Input representation:** `<s> Passage here. </s> Q: Question here? </s>`

Observations:
1. Decayed learning rates on finetuning seems to make it more robust (?)
2. 

|    bs32   |  :  |  bs48  | = |   1 : 1.5   |                   |
|-----------|-----|--------|---|-------------|-------------------|
|  lr2.5e-5 |  :  | lr3e-5 | = |   1 : 1.2   |  ≈ 1 : sqrt(1.5)  |


  

Table:

| steps | ep | bs |  lr | lr decay | Best F1 |   |
|:-----:|:--:|:--:|:---:|:--------:|:-------:|:-:|
|  5430 |  2 | 48 | 1.5 |    1.0   |  88.297 |   |
|       |    |    |     |          |         |   |
|  8144 |  2 | 32 | 1.5 |   0.75   |  88.562 |   |
|  8144 |  2 | 32 | 2.0 |   0.75   |  88.998 |   |
|  8144 |  2 | 32 | 2.5 |   0.75   |  89.477 |   |
|  8144 |  2 | 32 | 3.0 |   0.75   |  89.340 |   |
|  5430 |  2 | 48 | 2.5 |   0.75   |  89.229 |   |
|  5430 |  2 | 48 | 3.0 |   0.75   |  ***89.615*** | [Pytorch](https://drive.google.com/open?id=1pff390_zW4VJIcNXALbjP_yAerpH_V3l) |
|  5430 |  2 | 48 | 3.5 |   0.75   |  89.433 |   |
|  4073 |  2 | 64 | 3.5 |   0.75   |  89.144 |   |
|  5430 |  2.5 | 64 | 3.5 |   0.75   |  89.369 |   |
|       |    |    |     |          |         |   |
|   ??  | ?? | ?? |  ?? |    ??    |  **89.4** |  Official (dev set) |
|   ??  | ?? | ?? |  ?? |    ??    |  **89.795** |  Official (test set) |



## Experiment 1 (According to the original paper)
### Run on SQuAD 2.0 Dev Set

```c
lr_decay=1.0        
TOTAL_NUM_UPDATES=5430   # Number of training steps.
WARMUP_UPDATES=326       # Linearly increase LR over this many steps.
LR=1.5e-05               # Peak LR for fixed LR scheduler.
MAX_SENTENCES=3          # Batch size per GPU.
UPDATE_FREQ=2            # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=qa_records_squad_q
ROBERTA_PATH=roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.5 ./fairseq_train.py $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --task squad2 \
    --max-positions 512 \
    --arch roberta_qa_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion squad2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES \
    --lr_decay $lr_decay \
    --ddp-backend=no_c10d \
    --num-workers=0
```

```json
{
  "exact": 83.4329992419776,
  "f1": 86.7448817152165,
  "total": 11873,
  "HasAns_exact": 82.86099865047234,
  "HasAns_f1": 89.49426123562206,
  "HasAns_total": 5928,
  "NoAns_exact": 84.00336417157276,
  "NoAns_f1": 84.00336417157276,
  "NoAns_total": 5945,
  "best_exact": 85.21014065526826,
  "best_exact_thresh": -1.6142578125,
  "best_f1": 88.297090749954,
  "best_f1_thresh": -1.572265625
}
```





## Experiment 2 (Introduce LR Decay :D)
### Run on SQuAD 2.0 Dev Set
```c

lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144 
WARMUP_UPDATES=489   
LR=1.5e-05       
MAX_SENTENCES=4   
UPDATE_FREQ=1    
```

```json
{
  "exact": 84.03941716499621,
  "f1": 87.29093171231531,
  "total": 11873,
  "HasAns_exact": 83.02968960863697,
  "HasAns_f1": 89.54204322205142,
  "HasAns_total": 5928,
  "NoAns_exact": 85.04625735912532,
  "NoAns_f1": 85.04625735912532,
  "NoAns_total": 5945,
  "best_exact": 85.5217720879306,
  "best_exact_thresh": -1.921875,
  "best_f1": 88.56228638211618,
  "best_f1_thresh": -1.765625
}  
```









## Experiment 3 (Boost the learning rate to compensate the decayed LR!)
### Run on SQuAD 2.0 Dev Set
```c

lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144 
WARMUP_UPDATES=489 
LR=2e-05        
MAX_SENTENCES=4
UPDATE_FREQ=1    
```

```json
{
  "exact": 84.89850922260591,
  "f1": 88.04700949753051,
  "total": 11873,
  "HasAns_exact": 83.62010796221323,
  "HasAns_f1": 89.92613761204144,
  "HasAns_total": 5928,
  "NoAns_exact": 86.17325483599663,
  "NoAns_f1": 86.17325483599663,
  "NoAns_total": 5945,
  "best_exact": 86.07765518403099,
  "best_exact_thresh": -1.859375,
  "best_f1": 88.99848000856761,
  "best_f1_thresh": -1.611328125
}
```






## Experiment 4 (Further!)
### Run on SQuAD 2.0 Dev Set
```c
lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144 
WARMUP_UPDATES=489  
LR=2.5e-05      
MAX_SENTENCES=4 
UPDATE_FREQ=1 

 
```

```json
{
  "exact": 85.42070243409417,
  "f1": 88.5973743793479,
  "total": 11873,
  "HasAns_exact": 83.83940620782727,
  "HasAns_f1": 90.20185998751667,
  "HasAns_total": 5928,
  "NoAns_exact": 86.99747687132044,
  "NoAns_f1": 86.99747687132044,
  "NoAns_total": 5945,
  "best_exact": 86.41455403015244,
  "best_exact_thresh": -1.5517578125,
  "best_f1": 89.47730538540738,
  "best_f1_thresh": -1.328125
}
```





## Experiment 4 (Further..... too much! )
### Run on SQuAD 2.0 Dev Set
```c
lr_rate_decay=0.75
TOTAL_NUM_UPDATES=8144
WARMUP_UPDATES=489  
LR=3e-05          
MAX_SENTENCES=4   
UPDATE_FREQ=1     
```

```json
{
  "exact": 85.59757432830793,
  "f1": 88.73390560223615,
  "total": 11873,
  "HasAns_exact": 83.9574898785425,
  "HasAns_f1": 90.23914662877074,
  "HasAns_total": 5928,
  "NoAns_exact": 87.23296888141296,
  "NoAns_f1": 87.23296888141296,
  "NoAns_total": 5945,
  "best_exact": 86.33875178977512,
  "best_exact_thresh": -1.2626953125,
  "best_f1": 89.33994325354834,
  "best_f1_thresh": -1.259765625
}
```



## Experiment 5 (go back to bs48!)
### Run on SQuAD 2.0 Dev Set
```c
lr_rate_decay=0.75
TOTAL_NUM_UPDATES=5430 
WARMUP_UPDATES=326    
LR=3e-05            
MAX_SENTENCES=3     
UPDATE_FREQ=2     
```

```json
{
  "exact": 85.7997136359808,
  "f1": 88.8923704940676,
  "total": 11873,
  "HasAns_exact": 83.92375168690958,
  "HasAns_f1": 90.117934358311,
  "HasAns_total": 5928,
  "NoAns_exact": 87.67031118587047,
  "NoAns_f1": 87.67031118587047,
  "NoAns_total": 5945,
  "best_exact": 86.64196075128443,
  "best_exact_thresh": -1.15234375,
  "best_f1": 89.61546240072953,
  "best_f1_thresh": -1.15234375
}
```




## Experiment 6 (lower lr?)
### Run on SQuAD 2.0 Dev Set
```c
lr_rate_decay=0.75
TOTAL_NUM_UPDATES=5430 
WARMUP_UPDATES=326    
LR=2.5e-05            
MAX_SENTENCES=3     
UPDATE_FREQ=2     
```

```json
{
  "exact": 85.48808220331846,
  "f1": 88.58805430666887,
  "total": 11873,
  "HasAns_exact": 83.92375168690958,
  "HasAns_f1": 90.13258582710525,
  "HasAns_total": 5928,
  "NoAns_exact": 87.04793944491169,
  "NoAns_f1": 87.04793944491169,
  "NoAns_total": 5945,
  "best_exact": 86.30506190516297,
  "best_exact_thresh": -1.650390625,
  "best_f1": 89.22944509616022,
  "best_f1_thresh": -1.43359375
}
```



## Experiment 7 (how about higher? nah..)
### Run on SQuAD 2.0 Dev Set
```c
lr_rate_decay=0.75
TOTAL_NUM_UPDATES=5430 
WARMUP_UPDATES=326    
LR=3.5e-05            
MAX_SENTENCES=3     
UPDATE_FREQ=2     
```

```json
{
  "exact": 85.43754737640023,
  "f1": 88.63218801250815,
  "total": 11873,
  "HasAns_exact": 83.29959514170041,
  "HasAns_f1": 89.69803783274503,
  "HasAns_total": 5928,
  "NoAns_exact": 87.56938603868797,
  "NoAns_f1": 87.56938603868797,
  "NoAns_total": 5945,
  "best_exact": 86.3555967320812,
  "best_exact_thresh": -1.3154296875,
  "best_f1": 89.43337341665799,
  "best_f1_thresh": -1.28515625
}
```




