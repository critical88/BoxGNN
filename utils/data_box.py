import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import scipy.sparse as sp
from tqdm import tqdm

def _bi_norm_lap(adj):
            # D^{-1/2}AD^{-1/2}
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()

def _si_norm_lap(adj):
    # D^{-1}A
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

class TrainDataset(Dataset):
    def __init__(self, args, data, data_dict, data_stat):
        super(TrainDataset, self).__init__()
        self.data = data
        self.args = args
        self.num_neg = args.num_neg_sample

        self.user2item = data_dict["user2item"]
        self.n_items = data_stat["n_items"]
        self.all_items = range(self.n_items)
        # self.graph = np.array([graph.row, graph.col])
        
        
    def create_all_negative_sample(self):
        users = self.data[:,0]
        self.neg_items = self.negative_sampling(users, self.num_neg)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        d = self.data[idx]
        user = d[0]
        pos_item = d[1]
        # neg_item = self.negative_sampling([user], 1)[0]
        neg_item = self.neg_items[idx]
        # idx = np.where(self.graph[0] == int(user))[0]
        # histories = self.graph[:, idx]
        return user, pos_item, neg_item
    
    def negative_sampling(self, users, n):
        neg_items = []
        all_neg_items = np.random.choice(self.all_items, size=n * users.shape[0], replace=True).reshape( -1, n)
        for user, candidate_neg_items in zip(users, all_neg_items):
            user = int(user)
            negitems = []
            for negitem in candidate_neg_items:
                if negitem not in self.user2item[user]:
                    negitems.append(negitem)
            while(len(negitems) < n):
                while True:
                    negitem = np.random.choice(self.all_items, size=1)[0]
                    if negitem not in self.user2item[user]:
                        break
                negitems.append(negitem)
            
            neg_items.append(negitems)
        return neg_items


def train_collate(batch):
    users = []
    pos_items = []
    neg_items = []
    ## idx:0 users
    ## idx:1 items
    ## idx:2 neg_items
    ## idx:3 histories
    device = batch[0][0].device
    for idx, samples in enumerate(zip(*batch)):
        if idx == 0:
            users = torch.LongTensor(samples).to(device)
        elif idx == 1:
            pos_items = torch.LongTensor(samples).to(device)
        elif idx == 2:
            neg_items = torch.LongTensor(samples).to(device)
        # elif idx == 3:
            # samples = np.unique(np.concatenate(samples), axis=0)
            # histories = torch.LongTensor(samples).to(device)

        # users.append(user)
        # pos_items.append(pos_item)
        # neg_items.append(neg_item)
    return users, pos_items, neg_items
