
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_sum
from torch_scatter.utils import broadcast
from torch_scatter.composite import scatter_softmax
from collections import defaultdict

class Recommender(nn.Module):
    def __init__(self, args, data_stat, user_adj, item_adj):
        super(Recommender, self).__init__()

        self.n_users = data_stat['n_users']
        self.n_items = data_stat['n_items']
        self.n_tags = data_stat["n_tags"]
        self.n_nodes = data_stat['n_nodes']  # n_users + n_items

        self.decay = args.l2
        self.emb_size = args.dim
        self.context_hops = args.context_hops
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda \
                                                                      else torch.device("cpu")
        
        self.user_adj = self._convert_sp_mat_to_sp_tensor(user_adj).to_dense().to(self.device)
        self.item_adj = self._convert_sp_mat_to_sp_tensor(item_adj).to_dense().to(self.device)
        modules = []

        for _ in range(self.context_hops):
            linear = nn.Linear(self.n_tags, self.n_tags).to(self.device)
            nn.init.constant_(linear.bias, 0)
            modules.append(linear)
            modules.append(torch.nn.Tanh())
        self.networks = nn.ModuleList(modules)
        # self.all_embed = nn.Parameter(self.all_embed)



    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch):
        user, pos_item, neg_item = batch
        neg_item = torch.stack(neg_item).t().to(self.device)
        user_feature = self.user_adj[user]
        item_pos_feature = self.item_adj[pos_item]
        item_neg_feature = self.item_adj[neg_item.reshape(-1)]
        
        for net in self.networks:
            user_feature = net(user_feature)
            item_pos_feature = net(item_pos_feature)
            item_neg_feature = net(item_neg_feature)
        pos_score = torch.cosine_similarity(user_feature, item_pos_feature)
        neg_score = torch.cosine_similarity(user_feature, item_neg_feature)
        loss = -1 * torch.mean(nn.LogSigmoid()(pos_score - neg_score))
        return loss  , loss, None

    def generate(self):
        user_feature = self.user_adj
        item_feature= self.item_adj
        for net in self.networks:
            user_feature = net(user_feature)
            item_feature = net(item_feature)
        return user_feature, item_feature


    def rating(self, u_g_embeddings, i_g_embeddings, same_dim=False):
        return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2)
        # if same_dim:
        #     return torch.sum(u_g_embeddings * i_g_embeddings, dim=-1)
        # else:   
        #     return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_items = neg_items.reshape(batch_size, -1, self.emb_size)
        neg_scores = torch.sum(torch.mul(users.unsqueeze(1), neg_items), axis=2)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores.mean(axis=1)))

        return mf_loss 