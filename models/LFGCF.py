
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
    def __init__(self, channel, n_hops, n_users,n_items,n_tags, user_adj, item_adj, tag_adj):
        super(GraphConv, self).__init__()

        self.n_layers = n_hops
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags
        self.user_adj = user_adj
        self.item_adj = item_adj
        self.tag_adj = tag_adj

    def forward(self, user_emb, item_emb, tag_emb):

        """node dropout"""
        # all_res_emb = all_embed

        agg_final_user_emb = [user_emb]
        agg_final_item_emb = [item_emb]
        agg_final_tag_emb = [tag_emb]
        # entity_res_emb = entity_emb  # [n_entity, channel]
        # user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_layers):
            # all_embed = torch.cat([user_emb, entity_emb])
            user_emb = torch.sparse.mm(self.user_adj, tag_emb)
            item_emb = torch.sparse.mm(self.item_adj, tag_emb)
            tag_emb = torch.sparse.mm(self.tag_adj, torch.cat([user_emb, item_emb], dim=0))
            user_emb = F.normalize(user_emb)
            item_emb = F.normalize(item_emb)
            tag_emb = F.normalize(tag_emb)

            agg_final_user_emb.append(user_emb / (i+1))
            agg_final_item_emb.append(item_emb / (i+1))
            agg_final_tag_emb.append(tag_emb / (i+1))

        agg_final_user_emb = torch.stack(agg_final_user_emb).sum(dim=0)
        agg_final_item_emb = torch.stack(agg_final_item_emb).sum(dim=0)
        agg_final_tag_emb = torch.stack(agg_final_tag_emb).sum(dim=0)

        return agg_final_user_emb, agg_final_item_emb, agg_final_tag_emb


class Recommender(nn.Module):
    def __init__(self, args, data_stat, user_adj, item_adj, tag_adj):
        super(Recommender, self).__init__()

        self.n_users = data_stat['n_users']
        self.n_items = data_stat['n_items']
        self.n_tags = data_stat["n_tags"]
        self.n_nodes = data_stat['n_nodes']  # n_users + n_items + n_tags
        self.n_entities = data_stat['n_items'] + data_stat['n_tags']
        self.decay = args.l2
        self.emb_size = args.dim
        self.context_hops = args.context_hops
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda \
                                                                      else torch.device("cpu")
        self.user_adj = self._convert_sp_mat_to_sp_tensor(user_adj).to(self.device)
        self.item_adj = self._convert_sp_mat_to_sp_tensor(item_adj).to(self.device)
        self.tag_adj = self._convert_sp_mat_to_sp_tensor(tag_adj).to(self.device)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        # [n_users, n_items]

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_tags=self.n_tags,
                         user_adj=self.user_adj,
                         item_adj=self.item_adj,
                         tag_adj=self.tag_adj)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def transRT(self, user_emb, item_emb, tag_emb):
        return torch.norm(user_emb + item_emb - tag_emb, dim=-1).mean()

    def forward(self, batch):
        user, pos_item, neg_item, tag = batch
        neg_item = torch.stack(neg_item).t().to(self.device)
        user_emb, item_emb, tag_emb = torch.split(self.all_embed, [self.n_users, self.n_items, self.n_tags])
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb, tag_emb  = self.gcn(user_emb, item_emb, tag_emb)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item.reshape(-1)]
        tag_e = tag_emb[tag]
        batch_size = user.shape[0]
        regularizer = (torch.norm(user_emb[user]) ** 2
                       + torch.norm(item_gcn_emb[pos_item]) ** 2
                       + torch.norm(item_gcn_emb[neg_item.reshape(-1)]) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        rt_loss = self.transRT(u_e, pos_e, tag_e)
        loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss + emb_loss + 1e-5 * rt_loss, loss, emb_loss

    def generate(self):
        user_emb, item_emb, tag_emb = torch.split(self.all_embed, [self.n_users, self.n_items, self.n_tags])
        return self.gcn(user_emb,item_emb,tag_emb)[:2]

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

        return mf_loss 
