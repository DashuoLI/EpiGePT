import torch
import os
from pyfasta import Fasta
import numpy as np
import pandas as pd
from model_hg38 import EpiGePT
from model_hg38.config import *
from model_hg38.utils import *

os.environ['CUDA_VISIBLE_DEVICES']= '1' if torch.cuda.is_available() else '0'

model = EpiGePT.EpiGePT(WORD_NUM,TF_DIM,BATCH_SIZE)
model = load_weights(model,'pretrainModel/model.ckpt') # This already executed model.eval() to put model in inference mode, and called model.to(device) to get model ready for prediction

from pytorch_lightning.core.memory import ModelSummary
summary = ModelSummary(model, mode='top') # max_depth controls nesting level
print(summary)

# Simple example
print(f'Running simple example to validate model loading...')

SEQ_LENGTH = 128000
input_tf_feature = np.random.rand(1000, 711) # 711 TFs motif_score
input_seq_feature = np.zeros((1,4,SEQ_LENGTH)) #

print(input_seq_feature.shape)
print(input_tf_feature.shape)

predict = model_predict(model,input_seq_feature,input_tf_feature)
predict.shape # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)


# Actual prediction

from model_hg38.dataset import GenomicData
dataset = GenomicData([2])

from model_hg38.dataset import GenomicData
print("seq_embeds, tf_feats,targets_label,targets_mask")
print(len(dataset[0]))
seq_embeds = np.expand_dims(dataset[0][0].numpy(), axis=0)
tf_feats = dataset[0][1][:, :-1].numpy()
targets_label = dataset[0][2].numpy()
targets_mask = dataset[0][3].numpy()

print(input_seq_feature.shape)
print(input_tf_feature.shape)

print(seq_embeds.shape)
print(tf_feats.shape)


predict = model_predict(model, seq_embeds, tf_feats)
print(predict.shape) # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)
print(predict[0][0])
print(targets_label.shape)