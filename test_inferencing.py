import torch
import os
from pyfasta import Fasta
import numpy as np
import pandas as pd
from model_hg38 import EpiGePT
from model_hg38.config import *
from model_hg38.utils import *
import sys
os.environ['CUDA_VISIBLE_DEVICES']= '1' if torch.cuda.is_available() else '0'

# Load model
model_path = sys.argv[1]
model = EpiGePT.EpiGePT(WORD_NUM,TF_DIM,BATCH_SIZE)
model = load_weights(model,model_path) # This already executed model.eval() to put model in inference mode, and called model.to(device) to get model ready for prediction

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
test_cell_file = "/u/project/xjzhou/shuoli/methylation_foundation_model/test_cells.npy"
cell_type_array = np.load(test_cell_file)

from model_hg38.dataset import GenomicData
dataset = GenomicData(cell_type_array, disable_random = True)

print(dataset.__len__())
#print(dataset.signals)
#print(dataset.signals.shape)

#dataset_sample_idx = 2
#print("seq_embeds, tf_feats,targets_label,targets_mask")
#print(len(dataset[dataset_sample_idx]))

def predict_for_one_index(dataset_region_x_cellline_index, model):
	# Extract input features to be used in model_prediction method
	print('processing seq')
	seq_embeds = np.expand_dims(dataset[dataset_region_x_cellline_index][0].numpy(), axis=0)
	print('processing tf')
	tf_feats = dataset[dataset_region_x_cellline_index][1][:, :-1].numpy()
	targets_label = dataset[dataset_region_x_cellline_index][2].numpy()
	targets_mask = dataset[dataset_region_x_cellline_index][3].numpy()
	# END of data pre-processing

	# Validate data sanity
#	print(f'---=== targets label ===---')
#	print(targets_label)
#	print(targets_mask)

#	print(f'---=== Shape of example seq and tf features ===---')
#	print(input_seq_feature.shape)
#	print(input_tf_feature.shape)

#	print(f'---=== Shape of inference seq and tf features ===---')
#	print(seq_embeds.shape)
#	print(tf_feats.shape)
#	print(targets_label.shape)
#	print(targets_mask.shape)
	# END of data sanity check

	# Predict
	print('predicting')
	predict = model_predict(model, seq_embeds, tf_feats)

	# Check prediction
#	print(f'---=== Shape of prediction results ===---')
#	print(predict.shape) # (BATCH_SIZE, Number of bins, Number of epigenomic profiles)
	
	return predict[0], targets_label, targets_mask


#idx = 2

#print(f'---=== prediction results ===---')
#print(predict.shape)
#print(predict)
#print(predict[0][idx]) # {idx} row of prediction results
#print(f'---=== corresponding ground truth label ===---')
#print(targets_label.shape)
#print(targets_label)
#print(targets_label[idx])

def inverse_times10_log1p_transform_chunk(trans_chunk):
	arr_chunk = (np.expm1(trans_chunk)) / 10
	return arr_chunk

def inverse_odds_log1p_transform_chunk(trans_chunk):
	arr_chunk = 1 - np.exp(-trans_chunk)
	return arr_chunk

def evaluate_error_for_one_index(predict, targets_label, targets_mask, transform_type="odds_log1p"):
	if transform_type == "odds_log1p":
		unconverted_targets_label = inverse_odds_log1p_transform_chunk(targets_label)
		unconverted_predict = inverse_odds_log1p_transform_chunk(predict)
	elif transform_type == "times10_log1p":
		unconverted_targets_label = inverse_times10_log1p_transform_chunk(targets_label)
		unconverted_predict = inverse_times10_log1p_transform_chunk(predict)
#	print(f'---=== prediction results ===---')
#	print(unconverted_predict.shape)
#	print(unconverted_predict)
#	print(unconverted_predict[0][idx]) # {idx} row of prediction results
#	print(f'---=== corresponding ground truth label ===---')
#	print(unconverted_targets_label.shape)
#	print(unconverted_targets_label)
#	print(unconverted_targets_label[idx]) 	
	squared_error = np.sum((unconverted_predict - unconverted_targets_label) ** 2 * targets_mask, dtype = float)
	abs_error = np.sum(np.abs(unconverted_predict - unconverted_targets_label) * targets_mask, dtype = float)
	unmasked_count = np.sum(targets_mask, dtype = float)
#	print(squared_error, abs_error, unmasked_count, squared_error/unmasked_count, abs_error/unmasked_count)
	return squared_error, abs_error, unmasked_count


accumulated_sqerr = 0.0
accumulated_abserr = 0.0
accumulated_unmask_count = 0.0
for dataset_sample_idx in range(dataset.__len__()):
	predict, targets_label, targets_mask = predict_for_one_index(dataset_sample_idx, model)
	squared_error, abs_error, unmasked_count = evaluate_error_for_one_index(predict, targets_label, targets_mask, transform_type="odds_log1p")
	accumulated_sqerr += squared_error
	accumulated_abserr += abs_error
	accumulated_unmask_count += unmasked_count

print('MSE:', accumulated_sqerr/accumulated_unmask_count)
print('MAE:', accumulated_abserr/accumulated_unmask_count)
