import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)
from torch.utils.data import Dataset
from pyfasta import Fasta
from model_hg38.config import *
import random
import pandas as pd
import sys
from tqdm import tqdm
import os

acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}


class GenomicData(Dataset):
    def load_large_file(filename):
        blocksize = 1024  # tune this for performance/granularity
        try:
            mmap = np.load(filename, mmap_mode='r')
            ret = np.empty_like(mmap)
            n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
            for b in tqdm(range(n_blocks), desc=f"Loading {os.path.basename(filename)}"):
                ret[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
        finally:
            del mmap  # make sure file is closed again
        return ret

    def __init__(self,train_idx,path = 'EpiGePT',quantile_norm=False,isTrain = True):
        print('loading genome data...')
        self.geno_path = path
        self.genome = Fasta('%s/data/genome/hg38.fa'%path)

        print('loading encode data...')
        self.train_idx = train_idx

        print('loading motifscore...')
        self.np_tf_bs = np.load('%s/data/encode/motifscore_v1.npy'%path)
        # self.np_tf_bs = np.load('%s/data/encode/motifscore_v1.npy'%path, mmap_mode='r')
        # self.np_tf_bs = GenomicData.load_large_file(f'{path}/data/encode/motifscore_v1.npy')
        # print(f'Shape of loaded array: {self.np_tf_bs.shape}')

        pd_tf_gexp = pd.read_csv('%s/data/encode/aggregated_tf_expr.csv'%path,header=0,sep='\t',index_col=[0])
        pd_tf_gexp = pd_tf_gexp.T
        if quantile_norm:
            pd_tf_gexp = pd.DataFrame.transpose(self.quantile_norm_trans(pd.DataFrame.transpose(pd_tf_gexp)))
        self.pd_tf_gexp = np.log(pd_tf_gexp+1)
        
        print('loading targets...')
        self.signals = np.load(f'{path}/data/encode/targets_data_v1.npy')
        # self.signals = np.load(f'{path}/data/encode/targets_data_v1.npy', mmap_mode='r')
        # self.signals = GenomicData.load_large_file(f'{path}/data/encode/targets_data_v1.npy')
        # print(f'Shape of loaded array: {self.signals.shape}')

        print('loading mask...')
        self.mask_mat = np.load(f'{path}/data/encode/targets_mask_v1.npy')
        # self.mask_mat = np.load(f'{path}/data/encode/targets_mask_v1.npy', mmap_mode='r')
        # self.mask_mat = GenomicData.load_large_file(f'{path}/data/encode/targets_mask_v1.npy')
        # print(f'Shape of loaded array: {self.mask_mat.shape}')

        print('taking log of target data...')
        self.signals = np.log(self.signals + 1)
        self.regions = []
        # >overlap_count_gt50.128k.bin
        # origin
        print('loading bin data...')
        with open("%s/data/encode/overlap_count_gt50.128k.bin"%path, "rb") as file:
            lines = file.readlines()
        for line in lines:
            self.regions.append(line.decode('utf-8').split('\t')[:3])
            
        np.random.seed(1234)
        region_idx_subset = np.random.choice(np.arange(len(self.regions)),size=len(self.regions),replace=False)
        train_region_idx = np.random.choice(np.arange(len(region_idx_subset)),size=len(self.regions),replace=False)
        test_region_idx = [item for item in np.arange(len(region_idx_subset)) if item not in train_region_idx]
        if isTrain:
            self.region_idx_subset = region_idx_subset[train_region_idx]
        else:
            self.region_idx_subset = region_idx_subset[test_region_idx]
        self.celllines = self.pd_tf_gexp.index.values[train_idx]
        self.train_idx = train_idx
        print(train_idx)
        print(self.celllines,len(self.celllines))
    

        
    def quantile_norm_trans(self, matrix):
        rank_mean = matrix.stack().groupby(matrix.rank(method='first').stack().astype(int)).mean()
        return matrix.rank(method='min').stack().astype(int).map(rank_mean).unstack()

    def get_seq_from_meta(self,region_idx):
        seq_info = self.regions[region_idx]
        chrom = seq_info[0]
        start = int(seq_info[1])
        end = int(seq_info[2])
        seq = self.genome[chrom][start:end]
        return seq

    def seq2mat(self,seq):
        seq = seq.upper()
        h = 4
        w = len(seq)
        mat = np.zeros((h, w), dtype=bool)  # True or false in mat
        for i in range(w):
            if seq[i] != 'N':
                mat[acgt2num[seq[i]], i] = 1.
        return mat

    def get_signals(self,region_idx,cellline_idx):
        signals = self.signals[self.train_idx[cellline_idx],region_idx * 1000 : (region_idx + 1)*1000 ,:]
        mask = self.mask_mat[self.train_idx[cellline_idx],region_idx * 1000 : (region_idx + 1)*1000 ,:]
        return signals,mask
    
    def get_tf_state(self,region_idx,cellline_idx):
        tf_gexp_vec = self.pd_tf_gexp.loc[str(self.celllines[cellline_idx])].values 
        tf_gexp_feat = np.tile(tf_gexp_vec,(1000,1))

        tf_bs_feat = self.np_tf_bs[region_idx*1000:(region_idx+1)*1000]  
        return tf_bs_feat*tf_gexp_feat
    
        
    def __getitem__(self, index):
        region_idx = self.region_idx_subset[int(index // len(self.celllines))]
        cellline_idx = index % len(self.celllines) 
        seq = self.get_seq_from_meta(region_idx)
        seq_embeds = self.seq2mat(seq)
        tf_feats = self.get_tf_state(region_idx,cellline_idx)
    

        targets_label,targets_mask = self.get_signals(region_idx,cellline_idx)

        tf_feats = np.pad(tf_feats,((0, 0), (0, 1)),'constant',constant_values = (0,0))
        tf_feats = np.array(tf_feats,dtype='float16')
        
        seq_embeds = np.array(seq_embeds,dtype='float16')
        seq_embeds = torch.from_numpy(seq_embeds)
        tf_feats = torch.from_numpy(tf_feats)
        targets_label = torch.from_numpy(targets_label)
        targets_mask = torch.from_numpy(targets_mask)
        
        seq_embeds = seq_embeds.type(torch.FloatTensor)
        tf_feats = tf_feats.type(torch.FloatTensor)
        targets_label = targets_label.type(torch.FloatTensor)
        targets_mask = targets_mask.type(torch.FloatTensor)
        return (seq_embeds, tf_feats,targets_label,targets_mask)

    def __len__(self):
        return len(self.celllines)*len(self.region_idx_subset)
