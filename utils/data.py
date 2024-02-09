import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
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

class DataUtils():
    def __init__(self, args):

        super(DataUtils, self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{self.args.gpu_id}")

    def build_graph(self, train_data, user2tag, item2tag, data_stat):
        """
        build the user-tag-item graph whose edges are divided into three cate: user-item, user-tag, item-tag 
        """
        print("building graph")
        n_users = data_stat["n_users"]
        n_items = data_stat["n_items"]
        n_tags = data_stat["n_tags"]
        n_nodes = data_stat["n_nodes"]
        # users, items, tags = train_data["userID"].to_numpy(), user_tag_item["itemID"].to_numpy(), user_tag_item["tagID"].to_numpy()
        # user2item = np.unique(train_data[:, [0,1]],axis=0)
        # user2tag = np.unique(train_data[:, [0,2]],axis=0)
        # item2tag = np.unique(train_data[:, [1,2]],axis=0)

        ## offset tags
        
        train_data = train_data.copy()
        train_data[:,1] = train_data[:,1] + n_users
        
        
        row = train_data[:,0]
        col = train_data[:,1]
        graph_dict = {
            "none" : (row, col)
        }
        if user2tag is not None:
            user2tag = user2tag.copy()
            item2tag = item2tag.copy()
            
            user2tag[:,1] = user2tag[:,1] + n_users + n_items
            item2tag[:,1] = item2tag[:,1] + n_users + n_items
            
            ## offset items
            item2tag[:,0] = item2tag[:,0] + n_users
            
            graph_dict["user"] = (np.concatenate([row, user2tag[:,0]]), np.concatenate([col, user2tag[:,1]]))
            graph_dict["item"] = (np.concatenate([row, item2tag[:,0]]), np.concatenate([col, item2tag[:,1]]))
            graph_dict["all"] = (np.concatenate([row, user2tag[:,0], item2tag[:,0]]), np.concatenate([col, user2tag[:,1], item2tag[:,1]]))
            # if self.args.graph_type == "user":
            #     row = np.concatenate([row, user2tag[:,0]])
            #     col = np.concatenate([col, user2tag[:,1]])
            # elif self.args.graph_type == "item":
            #     row = np.concatenate([row, item2tag[:,0]])
            #     col = np.concatenate([col, item2tag[:,1]])
            # elif self.args.graph_type == "all":
            #     row = np.concatenate([row, user2tag[:,0], item2tag[:,0]])
            #     col = np.concatenate([col, user2tag[:,1], item2tag[:,1]])
            # elif self.args.graph_type == 'none':
            #     pass
            # else:
            #     raise Exception("unkonw graph type")
        # row = np.concatenate([user2item[:,0], user2tag[:,0], item2tag[:,0]])
        # col = np.concatenate([user2item[:,1], user2tag[:,1], item2tag[:,1]])
        cf_adjs = {}
        norm_mats = {}
        mean_mats = {}
        for k, g in graph_dict.items():
            row, col = g
            if self.args.inverse_r:
                row_t = np.concatenate([row, col])
                col_t = np.concatenate([col, row])
                row = row_t
                col = col_t
            idx = np.unique(np.stack([row, col]), axis=1)
            vals = [1.] * (idx.shape[1])
            
            cf_adj = sp.coo_matrix((vals, idx), shape=(n_nodes, n_nodes))
            # cf_adj = cf_adj + sp.eye(cf_adj.shape[0])
            norm_mat = _bi_norm_lap(cf_adj)
            mean_mat = _si_norm_lap(cf_adj)
            cf_adjs[k] = cf_adj
            norm_mats[k] = norm_mat
            mean_mats[k] = mean_mat

        return cf_adjs, norm_mats, mean_mats

    def build_interact_matrix(self, train_data, data_stat):
        """
        only considering the user-item pair info
        """
        print("building interact_matrix ...")

        n_users = data_stat["n_users"]
        n_items = data_stat["n_items"]
        cf = train_data[:,:2].copy()
        cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
        ## inverse
        row, col = cf[:, 0], cf[:, 1]
        if self.args.inverse_r:
            row = np.concatenate([cf[:, 0], cf[:, 1]])
            col = np.concatenate([cf[:, 1], cf[:, 0]])
        idx = np.unique(np.stack([row, col]), axis=1)
        vals = [1.] * (idx.shape[1])
        
        cf_adj = sp.coo_matrix((vals, idx), shape=(n_users + n_items, n_users + n_items))

        norm_mat = _bi_norm_lap(cf_adj)
        mean_mat = _si_norm_lap(cf_adj)
        
        return cf_adj, norm_mat, mean_mat
        # return cf_adj.tocsr()[:n_users, n_users:].tocoo(), norm_mat.tocsr()[:n_users, n_users:].tocoo(), mean_mat.tocsr()[:n_users, n_users:].tocoo()

    def read_files(self):
        print("reading files ...")
        data_path = os.path.join(self.args.data_path, self.args.dataset)
        train_file = os.path.join(data_path, "train.txt")
        val_file = os.path.join(data_path, "val.txt")
        test_file = os.path.join(data_path, "test.txt")
        user2tag_file = os.path.join(data_path, "user2tag.txt")
        item2tag_file = os.path.join(data_path, "item2tag.txt")
        
        train_data = pd.read_csv(train_file, sep="\t")[["userID", "itemID"]].drop_duplicates().reset_index(drop=True)
        val_data = pd.read_csv(val_file, sep="\t")
        test_data = pd.read_csv(test_file, sep="\t")

        user2tag, item2tag = None, None
        has_tag = os.path.exists(user2tag_file)
        if has_tag:
            user2tag = pd.read_csv(user2tag_file, sep="\t")
            item2tag = pd.read_csv(item2tag_file, sep="\t")
            user2tag, item2tag = user2tag.to_numpy(), item2tag.to_numpy()
        #### remove duplicates
        # user_tag_item = train_data 
        # train_data = train_data[["userID", "itemID"]].drop_duplicates()

        train_data, val_data, test_data = train_data.to_numpy(), val_data.to_numpy(), test_data.to_numpy()
        
        
        data_stat = self.__stat(train_data, val_data, test_data, user2tag, item2tag)

        user2item_dict = defaultdict(list)
        val_user2item = defaultdict(list)
        test_user2item = defaultdict(list)
        user2tag_dict = defaultdict(list)
        for idx in range(train_data.shape[0]):
            user2item_dict[train_data[idx, 0]].append(train_data[idx, 1])
        for idx in range(val_data.shape[0]):
            val_user2item[val_data[idx, 0]].append(val_data[idx, 1])
        for idx in range(test_data.shape[0]):
            test_user2item[test_data[idx, 0]].append(test_data[idx, 1])

        if has_tag:
            for idx in range(user2tag.shape[0]):
                user2tag_dict[user2tag[idx, 0]].append(user2tag[idx, 1])
        data_dict = {
            "user2item": user2item_dict,
            "user2tag": user2tag_dict,
            "val_user2item": val_user2item,
            "test_user2item": test_user2item
        }

        return train_data, val_data, test_data, user2tag, item2tag, data_dict, data_stat


    def __stat(self, train_data, val_data, test_data, user2tag, item2tag):
        ## first column is the userID, second column is the itemID
        n_users = max(max(max(train_data[:,0]), max(val_data[:,0])), max(test_data[:,0])) + 1
        n_items = max(max(max(train_data[:,1]), max(val_data[:,1])), max(test_data[:,1])) + 1
        n_tags = (max(user2tag[:,1]) + 1) if user2tag is not None else 0
        ## tag dont participate in the graph construction
        n_nodes = n_users + n_items + n_tags

        print(f"n_users:{n_users}")
        print(f"n_items:{n_items}")
        print(f"n_tags:{n_tags}")
        print(f"n_nodes:{n_nodes}")
        print(f"n_interaction:{len(train_data)}")
        if user2tag is not None:
            print(f"n_user2tag:{len(user2tag)}")
            print(f"n_item2tag:{len(item2tag)}")
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_tags": n_tags,
            "n_nodes": n_nodes
        }

class TrainDataset(Dataset):
    def __init__(self, args, data, data_dict, data_stat):
        super(TrainDataset, self).__init__()
        self.data = data
        self.args = args
        self.num_neg = args.num_neg_sample

        self.user2item = data_dict["user2item"]
        self.n_items = data_stat["n_items"]
        self.all_items = range(self.n_items)

        
        
    def create_all_negative_sample(self):
        print("prepare negative samples")
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
        return user, pos_item, neg_item
    
    def negative_sampling(self, users, n):
        # neg_items = []
        # for user in users:
        #     user = int(user)
            
        #     n_user_items = len(self.user2item[user])
        #     use_choice = (self.n_items - n_user_items) > 100000
        #     if use_choice:
        #         negitems = []
        #         for i in range(n):  # sample n times
        #             while True:
        #                     negitem = random.choice(self.all_items)
        #                     if negitem not in self.user2item[user]:
        #                         break
        #         negitems.append(negitem)
        #     else:
        #         user_unseen = list(set(self.all_items) - set(self.user2item[user]))

        #         negitems = random.sample(user_unseen, n)
        #     neg_items.append(negitems)
        
        # return neg_items

        neg_items = []
        all_neg_items = np.array(random.choices(self.all_items, k=n * users.shape[0])).reshape( -1, n)
        for user, candidate_neg_items in zip(users, all_neg_items):
            user = int(user)
            negitems = []
            for negitem in np.unique(candidate_neg_items):
                if negitem not in self.user2item[user]:
                    negitems.append(negitem)
            while(len(negitems) < n):
                while True:
                    negitem = random.choice(self.all_items)
                    if negitem not in self.user2item[user]:
                        break
                negitems.append(negitem)
            
            neg_items.append(negitems)
        return neg_items

class TestDataset(Dataset):

    def __init__(self, args, data):
        super(TestDataset, self).__init__()
        self.data = data
        self.args = args

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        d = self.data[idx]
        user = d[0]
        item = d[1]
        return user, pos

def train_collate(batch):
    users = []
    pos_items = []
    neg_items = []
    for (user, pos_item, neg_item) in batch:
        users.append(user)
        pos_items.append(pos_item)
        neg_items.append(neg_item)
    users = torch.cat(users)
    pos_items = torch.cat(pos_items)
    return users, pos_items, neg_items

def test_collate(batch):
    users = []
    pos_items = []
    for (user, pos_item) in batch:
        users.append(user)
        pos_items.append(pos_item)
    
    return users, items