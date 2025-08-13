import torch
import os
from pyfasta import Fasta
import numpy as np
import pandas as pd
from model_hg38 import EpiGePT
from model_hg38.config import *
from model_hg38.utils import *

os.environ['CUDA_VISIBLE_DEVICES']= '1' if torch.cuda.is_available() else '0'

# Load model
model = EpiGePT.EpiGePT(WORD_NUM,TF_DIM,BATCH_SIZE)
model = load_weights(model,'pretrainModel/model.ckpt') # This already executed model.eval() to put model in inference mode, and called model.to(device) to get model ready for prediction

# Show model structure
from pytorch_lightning.core.memory import ModelSummary
summary = ModelSummary(model, mode='top') # max_depth controls nesting level
print(summary)

# Simple example
print(f'---=== Running simple example to validate model loading... ===---')

SEQ_LENGTH = 128000
input_tf_feature = np.random.rand(1000, 711) # 711 TFs motif_score
input_seq_feature = np.zeros((1,4,SEQ_LENGTH)) #

print(input_seq_feature.shape)
print(input_tf_feature.shape)

predict = model_predict(model,input_seq_feature,input_tf_feature)
predict.shape # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)


# Actual prediction
cell_type_array = [2]
from model_hg38.dataset import GenomicData
dataset = GenomicData(cell_type_array, disable_random = True)

print(dataset.signals)
print(dataset.signals.shape)

dataset_sample_idx = 2
from model_hg38.dataset import GenomicData
print("seq_embeds, tf_feats,targets_label,targets_mask")
print(len(dataset[dataset_sample_idx]))

# Extract input features to be used in model_prediction method
print('processing seq')
seq_embeds = np.expand_dims(dataset[dataset_sample_idx][0].numpy(), axis=0)
print('processing tf')
tf_feats = dataset[dataset_sample_idx][1][:, :-1].numpy()
targets_label = dataset[dataset_sample_idx][2].numpy()
targets_mask = dataset[dataset_sample_idx][3].numpy()
# END of data pre-processing

print(f'---=== targets label ===---')
print(targets_label)
print(targets_mask)

print(f'---=== Shape of example seq and tf features ===---')
print(input_seq_feature.shape)
print(input_tf_feature.shape)

print(f'---=== Shape of inference seq and tf features ===---')
print(seq_embeds.shape)
print(tf_feats.shape)
print(targets_label.shape)
print(targets_mask.shape)

predict = model_predict(model, seq_embeds, tf_feats)

print(f'---=== Shape of prediction results ===---')
print(predict.shape) # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)

idx = 2
print(f'---=== {idx} row of prediction results ===---')
print(predict[0][idx])
print(f'---=== corresponding ground truth label ===---')
print(targets_label.shape)
print(targets_label[idx])
