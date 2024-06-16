
__author__ = "linfake"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_sum
from torch_scatter.utils import broadcast
from torch_scatter.composite import scatter_softmax
from collections import defaultdict





class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,n_entities,  graph, 
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.n_layers = n_hops
        self.graph = graph
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout


    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    

    def forward(self, user_emb, entity_emb,
                graph, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            graph = self._sparse_dropout(graph, self.node_dropout_rate)
        _indices = graph._indices()
        head, tail = _indices[0,:], _indices[1,:]
        all_embed = torch.cat([user_emb, entity_emb])
        # all_res_emb = all_embed

        agg_final_emb = [all_embed]
        # entity_res_emb = entity_emb  # [n_entity, channel]
        # user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_layers):
            # all_embed = torch.cat([user_emb, entity_emb])
            all_embed_agg = torch.sparse.mm(graph, all_embed)
            if mess_dropout:
                all_embed_agg = self.dropout(all_embed_agg)
            all_embed = all_embed_agg
            agg_final_emb.append(all_embed)
            # all_res_emb = all_embed + all_res_emb
            # user_emb, entity_emb = torch.split(all_embed_agg, [user_emb.shape[0], entity_emb.shape[0]])
            # entity_res_emb = entity_res_emb + entity_emb
            # user_res_emb = user_res_emb + user_emb

        agg_final_emb = torch.stack(agg_final_emb).sum(dim=0)

        user_res_emb, entity_res_emb = torch.split(agg_final_emb, [user_emb.shape[0], entity_emb.shape[0]])

        return user_res_emb, entity_res_emb
        # return torch.cat([entity_res_emb, entity_2nd_res_emb], dim=1), torch.cat([user_res_emb, user_2nd_res_emb], dim=1)


class Recommender(nn.Module):
    def __init__(self, args, data_stat, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_stat['n_users']
        self.n_items = data_stat['n_items']
        self.n_nodes = data_stat['n_nodes']  # n_users + n_items + n_tags
        self.n_entities = data_stat['n_items'] + data_stat['n_tags']
        self.decay = args.l2
        self.emb_size = args.dim
        self.context_hops = args.context_hops
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda \
                                                                      else torch.device("cpu")
        self.adj_mat = adj_mat['none']
        self.margin=0.7
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        # [n_users, n_items]
        self.graph = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_entities=self.n_entities,
                         graph=self.graph,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)



    def forward(self, batch):
        user, pos_item, neg_item = batch
        neg_item = torch.stack(neg_item).t().to(self.device)
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        user_gcn_emb, entity_gcn_emb  = self.gcn(user_emb,
                                                     entity_emb,
                                                     self.graph,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item.reshape(-1)]
        batch_size = user.shape[0]
        regularizer = (torch.norm(user_emb[user]) ** 2
                       + torch.norm(entity_emb[pos_item]) ** 2
                       + torch.norm(entity_emb[neg_item.reshape(-1)]) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss + emb_loss, loss, emb_loss

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        entity_emb,
                        self.graph,
                        mess_dropout=False, node_dropout=False)[:2]

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
