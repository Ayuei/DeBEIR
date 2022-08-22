#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from functools import partial

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
import sys
from sentence_transformers.evaluation import TripletEvaluator
from wandb import wandb
import torch_optimizer as optim

from nir.training.utils import get_scheduler_with_wandb

from torch.utils.data import DataLoader

sys.path.append("/home/vin/Projects/nir/")


# In[2]:


from nir.training.train_sentence_encoder import train_huggingface_transformer
from nir.training.utils import DatasetToSentTrans, TokenizerOverload, LoggingLoss, LoggingEvaluator

TASK_NAME = "contrastive_metamap_marco_passage"
OUTPUT_DIR = f"./outputs/biencoder/{TASK_NAME}/"
DATA_DIR = "../data/"
BATCH_SIZE = 50 

os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[3]:


from transformers import AutoTokenizer
import datasets
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample

dataset = datasets.load_dataset("csv", data_files={"train": "/home/vin/Datasets/marco-passage-ranking/med_marco_train_enriched.csv"})


def split_k_fold(n_fold, data_files):
    percentage = 100//n_fold
    
    vals_ds = datasets.load_dataset('csv', split=[
    f'train[{k}%:{k+percentage}%]' for k in range(0, 100, percentage)
    ], data_files=data_files)
    
    trains_ds = datasets.load_dataset('csv', split=[
    f'train[:{k}%]+train[{k+percentage}%:]' for k in range(0, 100, percentage)
    ], data_files=data_files)
    
    return trains_ds, vals_ds

dataset = split_k_fold(5, data_files={"train": "/home/vin/Datasets/marco-passage-ranking/med_marco_train_enriched.csv"})

train_fold1 = dataset[0][0]
val_fold1 = dataset[1][0]

train_dataset = DatasetToSentTrans(train_fold1, text_cols=['q_text', 'pos', 'neg'])
val_dataset = DatasetToSentTrans(val_fold1, text_cols=['q_text', 'pos', 'neg'])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

encoder = SentenceTransformer("./Bio_ClinicalBERT")

wandb.init()
wandb.watch(encoder)

evaluator = LoggingEvaluator(TripletEvaluator.from_input_examples(val_dataset), wandb)
train_loss = LoggingLoss(losses.TripletLoss(model=encoder), wandb)

encoder._get_scheduler = partial(get_scheduler_with_wandb, wandb)

warmup_steps = 0.1 * len(train_dataloader)

encoder.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=6,
          evaluator=evaluator,
          warmup_steps=5,
          evaluation_steps=500,
          use_amp=True,
          optimizer_class=optim.Yogi,
          checkpoint_path=OUTPUT_DIR,
          optimizer_params = {'lr': 1e-4},
          scheduler = "warmupcosine"
)
