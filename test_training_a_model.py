# additional data needed for training
#	new file:   EpiGePT/data/encode/aggregated_tf_expr.csv
#	new file:   EpiGePT/data/encode/motifscore_v1.npy
#	new file:   EpiGePT/data/encode/overlap_count_gt50.128k.bin
#	new file:   EpiGePT/data/encode/targets_data_v1.npy
#	new file:   EpiGePT/data/encode/targets_mask_v1.npy



import torch
import os
from pyfasta import Fasta
import numpy as np
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']='1'
from model_hg38 import EpiGePT
from model_hg38.config import *
from model_hg38.utils import *

#training
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model.config import *

print(torch.cuda.is_available())

# Loading model
model = EpiGePT.EpiGePT(WORD_NUM,TF_DIM,BATCH_SIZE)
model = load_weights(model,'pretrainModel/model.ckpt')

# Testing model prediction
#SEQ_LENGTH = 128000
#input_tf_feature = np.random.rand(1000, 711) # 711 TFs\n
#input_seq_feature = np.zeros((1,4,SEQ_LENGTH))
#predict = model_predict(model,input_seq_feature,input_tf_feature)

#print("Prediction results")
#print(predict.shape) # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)

# Testing incremental training
trainer = pl.Trainer(
	max_epochs=90,
	logger=pl_loggers.TensorBoardLogger(save_dir='logs', name='TensorBoard', version=5),
	callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3)],
	default_root_dir=os.getcwd(),
	gpus = 1,
	)
trainer.fit(model)
