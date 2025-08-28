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
from model_hg38 import EpiGePT
from model_hg38.config import *
from model_hg38.utils import *

#training
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

import sys
from datetime import datetime

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES']= '1' if torch.cuda.is_available() else '0'
from model.config import *

print(torch.cuda.is_available())
print(os.environ['CUDA_VISIBLE_DEVICES'])

# Loading model
#model_checkpoint = sys.argv[1]
model = EpiGePT.EpiGePT(WORD_NUM,TF_DIM,BATCH_SIZE)
#model = load_weights(model,'pretrainModel/model.ckpt')
#model = load_weights(model,model_checkpoint)

#for name, param in model.named_parameters():
#	print(name, param.requires_grad)

# Testing model prediction
#SEQ_LENGTH = 128000
#input_tf_feature = np.random.rand(1000, 711) # 711 TFs\n
#input_seq_feature = np.zeros((1,4,SEQ_LENGTH))
#predict = model_predict(model,input_seq_feature,input_tf_feature)

#print("Prediction results")
#print(predict.shape) # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)

# Testing incremental training
if __name__ == '__main__':
	val_every_n_epochs = 1
	date_str = datetime.now().strftime("%Y-%m-%d")
	model_checkpoint = sys.argv[1]
	checkpoints_path = sys.argv[2] #"/u/project/xjzhou/shuoli/methylation_foundation_model/EpiGePT_times10/training_checkpoint/times10_allParamsTrainable"
	version = int(sys.argv[3])
	is_frozen = sys.argv[4]
	model = load_weights(model,model_checkpoint)
	model.train()
	if is_frozen == "TRUE":
		max_epochs = 5
		for param in model.convmodule.parameters(): # Freeze convmodule
			param.requires_grad = False
		for param in model.encoder.parameters(): # Freeze encoder
			param.requires_grad = False
	else:
		max_epochs = 2
	checkpoint_callback = ModelCheckpoint(
		dirpath=checkpoints_path, 
		filename=f"fa_classifier_{date_str}_{{epoch:02d}}",
		period=val_every_n_epochs,
		save_top_k=-1,  # <--- this is important!
	)
	if torch.cuda.is_available():
		trainer = pl.Trainer(
			max_epochs=max_epochs,
			logger=pl_loggers.TensorBoardLogger(save_dir='logs', name='TensorBoard', version=version),
			callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3), checkpoint_callback],
			default_root_dir=os.getcwd(),
			gpus = 1,
			)
	else:
		trainer = pl.Trainer(
			max_epochs=max_epochs,
			logger=pl_loggers.TensorBoardLogger(save_dir='logs', name='TensorBoard', version=version),
			callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3), checkpoint_callback],
			default_root_dir=os.getcwd(),
			# gpus = 1,
			)

	trainer.fit(model)
