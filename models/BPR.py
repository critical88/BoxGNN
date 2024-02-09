
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
    def __init__(self, args, data_stat):
        super(Recommender, self).__init__()

        self.n_users = data_stat['n_users']
        self.n_items = data_stat['n_items']
        self.n_nodes = data_stat['n_nodes']  # n_users + n_items

        self.decay = args.l2
        self.emb_size = args.dim
        self.context_hops = args.context_hops
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda \
                                                                      else torch.device("cpu")
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        # [n_users, n_items]


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)



    def forward(self, batch):
        user, pos_item, neg_item = batch
        neg_item = torch.stack(neg_item).t().to(self.device)
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        u_e = user_emb[user]
        pos_e, neg_e = item_emb[pos_item], item_emb[neg_item.reshape(-1)]
        batch_size = user.shape[0]
        regularizer = (torch.norm(user_emb[user]) ** 2
                       + torch.norm(item_emb[pos_item]) ** 2
                       + torch.norm(item_emb[neg_item.reshape(-1)]) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss + emb_loss, loss, emb_loss

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return user_emb, item_emb


    def rating(self, u_g_embeddings, i_g_embeddings, same_dim=False):
        # return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2)
        if same_dim:
            return torch.sum(u_g_embeddings * i_g_embeddings, dim=-1)
        else:   
            return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_items = neg_items.reshape(batch_size, -1, self.emb_size)
        neg_scores = torch.sum(torch.mul(users.unsqueeze(1), neg_items), axis=2)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores.mean(axis=1)))

        # pos_score = torch.cosine_similarity(users, pos_items)
        
        # neg_score = torch.cosine_similarity(users.unsqueeze(1), neg_items.view(batch_size, -1, self.emb_size), dim=2)
        # neg_score = F.relu(neg_score - self.margin)

        # mf_loss = (1 - pos_score) +  torch.sum(neg_score, dim=-1) / (torch.sum(neg_score > 0, dim=-1) + 1e-5)
        # mf_loss = torch.mean(mf_loss)


        return mf_loss 