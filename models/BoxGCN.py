'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch_scatter import scatter_sum, scatter_mean, scatter_min,scatter_max
from torch_scatter.composite import scatter_softmax
import math

class MultilayerNN(nn.Module):
    # torch min will just return a single element in the tensor
    def __init__(self, center_dim, offset_dim):
        super(MultilayerNN, self).__init__()
        self.center_dim = center_dim
        self.offset_dim = offset_dim

        expand_dim = center_dim * 2

        self.mats1_center = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_center)
        self.register_parameter("mats1_center", self.mats1_center)

        self.mats1_offset = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_offset)
        self.register_parameter("mats1_offset", self.mats1_offset)

        self.post_mats_center = nn.Parameter(torch.FloatTensor(expand_dim * 2, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_center)
        self.register_parameter("post_mats_center", self.post_mats_center)

        self.post_mats_offset = nn.Parameter(torch.FloatTensor(expand_dim * 2, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_offset)
        self.register_parameter("post_mats_offset", self.post_mats_offset)

    def forward(self, center_emb, offset_emb):
        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        temp1 = F.relu(torch.matmul(center_emb, self.mats1_center))
        temp2 = F.relu(torch.matmul(offset_emb, self.mats1_offset))

        temp3 = torch.cat([temp1, temp2], dim=-1)

        out_center = torch.matmul(temp3, self.post_mats_center)
        out_offset = torch.matmul(temp3, self.post_mats_offset)

        return out_center, out_offset


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, idx, dim_size):
        
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        layer2_act = self.layer2(layer1_act)
        attention = scatter_softmax(src=layer2_act, index=idx, dim=0)
        user_embedding = scatter_sum(attention * embeddings, index=idx, dim=0, dim_size=dim_size)
        # embedding = torch.sum(attention * embeddings, dim=0)
        
        # user_embedding = scatter_mean(embeddings, index=idx, dim=0, dim_size=dim_size)
        return user_embedding

class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, idx, dim_size, use_min=True):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = scatter_mean(src=layer1_act, index=idx, dim=0, dim_size=dim_size)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        if use_min:
            offset = scatter_min(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]
        else:
            offset = scatter_max(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]
        return offset * gate

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, emb_size, n_hops, n_users, n_items, n_entities, device,
                user2tag, item2tag,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.n_layers = n_hops
        self.emb_size = emb_size
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device

        self.user2tag = user2tag
        self.item2tag = item2tag.clone()
        self.item2tag[:,1] = item2tag[:,0]
        self.item2tag[:,0] = item2tag[:,1]
        self.item2tag = torch.cat([item2tag, self.item2tag], dim=0)

        idx = torch.arange(self.n_users).to(self.device)
        self.user_union_idx = torch.cat([idx, idx], dim=0)

        self.center_net = CenterIntersection(self.emb_size)
        self.offset_net = BoxOffsetIntersection(self.emb_size)
        self.trans_nn = MultilayerNN(self.emb_size, self.emb_size)

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

    def user_union(self, embs1, offset1, embs2, offset2):
        embs = torch.cat([embs1, embs2], dim=0)
        offset = torch.cat([offset1, offset2], dim=0)
        agg_emb = self.center_net(embs, self.user_union_idx, embs1.shape[0])
        ent_offset_emb = F.relu(self.offset_net(offset, self.user_union_idx, embs1.shape[0], use_min=False))
        
        return agg_emb, ent_offset_emb

    def intersection(self):
        pass

    def forward(self, user_emb, user_offset_emb, item_emb, item_offset_emb, graph, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            graph = self._sparse_dropout(graph, self.node_dropout_rate)
        _indices = graph._indices()

        head, tail = _indices[0,:], _indices[1,:]
        # head, tail = self.item2tag[:,0], self.item2tag[:,1]
        ## all interaction is positive
        # i2u_interacted_rel = torch.LongTensor([0]).to(self.device)
        # i2u_rel_emb = relation_embedding[i2u_interacted_rel, :]
        # i2u_rel_offset = relation_offset_embedding[i2u_interacted_rel, :]
        
        # u2i_interact_rel = torch.LongTensor([1]).to(self.device)
        # u2i_rel_emb = relation_embedding[u2i_interact_rel, :]
        # u2i_rel_offset = relation_offset_embedding[u2i_interact_rel, :]

        all_embs = torch.cat([user_emb, item_emb], dim=0)
        all_offset_emb = F.relu(torch.cat([user_offset_emb, item_offset_emb], dim=0))
        
        # user_offset_emb = torch.zeros([self.n_users, self.emb_size]).to(self.device)
        # item_offset_emb = torch.zeros([self.n_entities, self.emb_size]).to(self.device)

        agg_layer_emb = [all_embs]
        agg_layer_offset = [all_offset_emb]

        # user_final_emb = [user_emb]
        # item_final_emb = [item_emb]
        # user_final_offset = [user_offset_emb]
        # item_final_offset = [item_offset_emb]

        for _ in range(self.n_layers):
            ## interaction
            history_embs = all_embs[tail]
            agg_emb = self.center_net(history_embs, head, self.n_entities + self.n_users)
            
            
            inter_user_ordinal = (tail >= self.n_users) & (head < self.n_users) & (tail < self.n_users + self.n_items)
            # inter_history_embs = all_embs[tail[inter_user_ordinal]]
            # inter_agg_emb = self.center_net(inter_history_embs, head[inter_user_ordinal], self.n_entities + self.n_users)
            # inter_user_agg, inter_item_agg,_ = torch.split(inter_agg_emb, [self.n_users, self.n_items, self.n_entities - self.n_items])
            
            inter_user_history_offset = F.relu(all_offset_emb[tail[inter_user_ordinal]])
            inter_user_offset_emb = self.offset_net(inter_user_history_offset, head[inter_user_ordinal], self.n_users + self.n_entities, use_min=True)
            inter_user_offset_emb = inter_user_offset_emb[:self.n_users]
            
            ### user-tag
            # tag_user_ordinal = (head < self.n_users) & (tail >= self.n_users + self.n_items)
            user_tag_ordinal = (head < self.n_users) & (tail >= self.n_users + self.n_items)

            # ut_embs = all_embs[tail[user_tag_ordinal ]]

            # ut_agg_emb = self.center_net(ut_embs, head[user_tag_ordinal ], self.n_entities + self.n_users)

            # ut_user_agg, _, ut_tag_agg = torch.split(ut_agg_emb, [self.n_users, self.n_items, self.n_entities - self.n_items])

            user_tag_history_offset = F.relu(all_offset_emb[tail[user_tag_ordinal]])
            # tag_user_history_offset = F.relu(all_offset_emb[tail[tag_user_ordinal]])
            ut_user_offset_emb = self.offset_net(user_tag_history_offset, head[user_tag_ordinal], self.n_users + self.n_entities, use_min=False)
            ut_user_offset_emb = ut_user_offset_emb[:self.n_users]
            
            # ut_tag_offset_emb = self.offset_net(tag_user_history_offset, head[tag_user_ordinal], self.n_users + self.n_entities, use_min=True)
            # ut_tag_offset_emb = ut_tag_offset_emb[self.n_users+self.n_items:]

            ### item-tag
            # tag_item_ordinal = (head >= self.n_users) & (head < self.n_users + self.n_items) & (tail >= self.n_users + self.n_items)
            # item_tag_ordinal = (head >= self.n_users + self.n_items) & (tail >= self.n_users) & (tail < self.n_users + self.n_items)

            # ut_embs = all_embs[tail[tag_item_ordinal | item_tag_ordinal]]

            # it_agg_emb = self.center_net(ut_embs, head[tag_item_ordinal | item_tag_ordinal], self.n_entities + self.n_users)

            # _,it_item_agg, it_tag_agg = torch.split(it_agg_emb, [self.n_users, self.n_items, self.n_entities - self.n_items])

            # item_tag_history_offset = F.relu(all_offset_emb[tail[item_tag_ordinal]])
            # tag_item_history_offset = F.relu(all_offset_emb[tail[tag_item_ordinal]])
            # it_item_offset_emb = self.offset_net(item_tag_history_offset, head[item_tag_ordinal], self.n_users + self.n_entities, use_min=False)
            # it_item_offset_emb = it_item_offset_emb[self.n_users:self.n_users + self.n_items]
            
            # it_tag_offset_emb = self.offset_net(tag_item_history_offset, head[tag_item_ordinal], self.n_users + self.n_entities, use_min=True)
            # it_tag_offset_emb = it_tag_offset_emb[self.n_users+self.n_items:]
            
            ### item
            item_ordinal = (head >= self.n_users) & (head < self.n_users + self.n_items)

            # item_history_embs = all_embs[tail[item_ordinal]]
            # item_agg = self.center_net(item_history_embs, head[item_ordinal], self.n_entities + self.n_users)
            # item_agg = item_agg[self.n_users:self.n_items + self.n_users]
            item_history_offset = F.relu(all_offset_emb[tail[item_ordinal]])
            item_offset = self.offset_net(item_history_offset, head[item_ordinal], self.n_users + self.n_entities, use_min=False)
            item_offset = item_offset[self.n_users:self.n_users + self.n_items]
            
            ### tag
            tag_ordinal = (head >= self.n_users + self.n_items) 

            # tag_history_embs = all_embs[tail[tag_ordinal]]
            # tag_agg = self.center_net(tag_history_embs, head[tag_ordinal], self.n_entities + self.n_users)
            # tag_agg = tag_agg[self.n_items + self.n_users:]
            tag_history_offset = F.relu(all_offset_emb[tail[tag_ordinal]])
            tag_offset = self.offset_net(tag_history_offset, head[tag_ordinal], self.n_users + self.n_entities, use_min=True)
            tag_offset = tag_offset[self.n_users + self.n_items:]

            user_offset = torch.cat([inter_user_offset_emb, ut_user_offset_emb], dim=0)
            user_offset = F.relu(self.offset_net(user_offset, self.user_union_idx, inter_user_offset_emb.shape[0], use_min=False))
            # user_agg, user_offset = self.user_union(inter_user_agg, inter_user_offset_emb, ut_user_agg, ut_user_offset_emb)
            # item_agg, item_offset = self.union(inter_item_agg, inter_item_offset_emb, it_item_agg, it_item_offset_emb)
            # tag_agg, tag_offset = self.union(ut_tag_agg, ut_tag_offset_emb, it_tag_agg, it_tag_offset_emb)

            agg_offset_emb = torch.cat([user_offset, item_offset, tag_offset], dim=0)
            # ent_agg_emb = self.center_net(all_embs[self.item2tag[:,1]], self.item2tag[:,0], self.n_entities + self.n_users)
            # item_agg_emb = self.center_net(all_embs[self.item2tag[:,1]], self.item2tag[:,0], self.n_entities + self.n_users)
            # tag_agg_emb = self.center_net(all_embs[self.item2tag[:,0]], self.item2tag[:,1], self.n_entities + self.n_users)
            # item_offset_emb = self.offset_net(all_embs[self.item2tag[:,1]], self.item2tag[:,0], self.n_entities + self.n_users, use_min=False)
            # tag_offset_emb = self.offset_net(all_embs[self.item2tag[:,0]], self.item2tag[:,1], self.n_entities + self.n_users, use_min=True)
            # item_offset_emb = self.offset_net(item_history_offset, head[head >= self.n_users], self.n_users + self.n_entities, use_min=False )
            
            
            # tag_offset_emb = self.offset_net(history_offset, head, self.n_users + self.n_entities, use_min=True)
            # tag_offset_emb = tag_offset_emb[self.n_users + self.n_items:]

            # agg_emb = torch.cat([user_agg, ent_agg_emb[self.n_users:]], dim=0)
            # agg_offset_emb = ent_offset_emb
            # agg_offset_emb = torch.cat([ent_offset_emb, tag_offset_emb], dim=0)
            # agg_emb = torch.cat([agg_emb[:self.n_users], item_agg_emb[self.n_users:self.n_users+self.n_items], tag_agg_emb[self.n_users + self.n_items:]], dim=0)
            # agg_offset_emb = torch.cat([user_offset_emb[:self.n_users], item_offset_emb[self.n_users:self.n_users+self.n_items], tag_offset_emb[self.n_users + self.n_items:]], dim=0)

            agg_emb = F.normalize(agg_emb)

            # agg_emb, agg_offset_emb = self.trans_nn(agg_emb, agg_offset_emb)
            agg_layer_emb.append(agg_emb)
            agg_layer_offset.append(agg_offset_emb)
            
            all_embs = agg_emb
            all_offset_emb = agg_offset_emb

            # history_i2u_embs = item_emb[items] # + i2u_rel_emb.expand(items.shape[0], self.emb_size)
            # history_i2u_offset =  F.relu(item_offset_emb[items]) # + i2u_rel_offset.expand(items.shape[0], self.emb_size)) #F.relu(item_offset_emb[items]) 
            # user_agg_emb = self.center_net(history_i2u_embs, users, self.n_users)
            # user_agg_offset_emb = self.offset_net(history_i2u_offset, users, self.n_users)
            # # user_emb, user_offset_emb = self.trans_nn(user_emb, user_offset_emb)
            
            # # user_agg_emb = user_agg_emb  + u2i_rel_emb.expand(user_emb.shape[0], self.emb_size)
            # # user_agg_offset_emb = F.relu(user_agg_offset_emb + u2i_rel_offset.expand(user_offset_emb.shape[0], self.emb_size))
        
            
            # history_u2i_embs = user_emb[users] # + u2i_rel_emb.expand(users.shape[0], self.emb_size)
            # history_u2i_offset = F.relu(user_offset_emb[users]) # + u2i_rel_offset.expand(users.shape[0], self.emb_size)) # F.relu(user_offset_emb[users]) 
            # item_agg_emb = self.center_net(history_u2i_embs, items, self.n_entities)
            # item_agg_offset_emb = self.offset_net(history_u2i_offset, items, self.n_entities)
            # # item_emb, item_offset_emb = self.trans_nn(item_emb, item_offset_emb)

            # # item_agg_emb = item_agg_emb + i2u_rel_emb.expand(item_emb.shape[0], self.emb_size)
            # # item_agg_offset_emb = F.relu(item_agg_offset_emb + i2u_rel_offset.expand(item_emb.shape[0], self.emb_size))
            
            # user_final_emb.append(user_agg_emb)
            # item_final_emb.append(item_agg_emb)
            # user_final_offset.append(user_agg_offset_emb)
            # item_final_offset.append(item_agg_offset_emb)

            # user_emb = user_agg_emb
            # item_emb = item_agg_emb
            # user_offset_emb = user_agg_offset_emb
            # item_offset_emb = item_agg_offset_emb
            # item_final_emb = item_emb
        
        agg_final_emb = agg_layer_emb[-1]#torch.stack(agg_layer_emb).sum(dim=0)
        agg_final_offset = agg_layer_offset[-1]#torch.stack(agg_layer_offset).sum(dim=0)

        user_final_emb, item_final_emb = torch.split(agg_final_emb, [self.n_users, self.n_entities])

        user_final_offset, item_final_offset = torch.split(agg_final_offset, [self.n_users, self.n_entities])

        # item_final_emb, item_final_offset = item_emb, item_offset_emb
        # item_final_offset = agg_final_offset[self.n_users:, :]

        # user_final_emb = torch.stack(user_final_emb).mean(dim=0)
        # user_final_offset = torch.stack(user_final_offset).mean(dim=0)
        # item_final_emb = torch.stack(item_final_emb).mean(dim=0)
        # item_final_offset = torch.stack(item_final_offset).mean(dim=0)

        # user_final_emb = user_final_emb + u2i_rel_emb.expand(user_emb.shape[0], self.emb_size)
        # user_final_offset = F.relu(user_final_offset + u2i_rel_offset.expand(user_offset_emb.shape[0], self.emb_size))
        

        return user_final_emb, user_final_offset, item_final_emb, item_final_offset


class Recommender(nn.Module):
    def __init__(self, args, data_stat, adj_mat, user2tag, item2tag):
        super(Recommender, self).__init__()

        self.n_users = data_stat['n_users']
        self.n_items = data_stat['n_items']
        self.n_nodes = data_stat['n_nodes']  # n_users + n_items + n_tags
        self.n_entities = data_stat['n_items'] + data_stat['n_tags']
        self.decay = args.l2
        self.emb_size = args.dim
        self.n_layers = args.context_hops
        self.cen = 0.02
        self.margin=0.7
        self.logit_cal = args.logit_cal
        self.vol_coeff = args.vol_coeff
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.contrastive_loss = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda \
                                                                      else torch.device("cpu")
        
        self.gamma = args.gamma

        # user2tag[:,1] += self.n_users + self.n_items
        item2tag[:,1] += self.n_items + self.n_users
        item2tag[:,0] += self.n_users
        self.user2tag = torch.LongTensor(user2tag).to(self.device)
        self.item2tag = torch.LongTensor(item2tag).to(self.device)
        self.adj_mat = adj_mat[args.graph_type]
        self.adj_mat_none = adj_mat['none']
        
        self._init_weight()
        self.tau = args.tau
        self.gcn = self._init_model()
        

    def _init_model(self):
        return GraphConv(emb_size=self.emb_size,
                         n_hops=self.n_layers,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_entities=self.n_entities,
                         device=self.device,
                         user2tag=self.user2tag,
                         item2tag=self.item2tag,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_users + self.n_entities, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)
        self.all_offset = initializer(torch.empty([self.n_users + self.n_entities, self.emb_size])) 
        self.all_offset = nn.Parameter(self.all_offset)

        # [n_users, n_items]
        self.graph = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.graph_none = self._convert_sp_mat_to_sp_tensor(self.adj_mat_none).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def log_inter_volumes(self, inter_min, inter_max, scale=1.):
        eps = 1e-16
        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        log_vol = torch.sum(
            torch.log(
                F.softplus(inter_max - inter_min, beta=0.7).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)
        return log_vol
        
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def infoNce(self, sim, batchsz):
        N = 2 * batchsz

        sim = sim / self.tau

        sim_i_j = torch.diag(sim, batchsz)
        sim_j_i = torch.diag(sim, -batchsz)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batchsz)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.contrastive_loss(logits, labels)
        return loss

    def cal_logit_vol(self, box1_embs, box1_off, box2_embs, box2_off):
        box1_min = (box1_embs - F.relu(box1_off))
        box1_max = (box1_embs + F.relu(box1_off))
        box2_min = (box2_embs - F.relu(box2_off))
        box2_max = (box2_embs + F.relu(box2_off))

        inter_min = torch.max(box1_min, box2_min)
        inter_max = torch.min(box1_max, box2_max)
        cen_dis = torch.norm(box1_embs - box2_embs, p=2, dim=-1)
        outer_min = torch.min(box1_min, box2_min)
        outer_max = torch.max(box1_max, box2_max)
        inter_vol = self.log_inter_volumes(inter_min, inter_max)
        c2 = torch.norm(outer_max - outer_min, p=2, dim=-1)
        d_loss = torch.sqrt(cen_dis) / torch.sqrt(c2)
        logit = torch.sigmoid(inter_vol-2) - d_loss
        logit[logit.isnan()] = 0
        return logit
    def cal_logit_box(self, user_center_embedding, user_offset_embedding, item_center_embedding, item_offset_embedding, training=True):
        if self.logit_cal == "box":
            delta = (item_center_embedding - user_center_embedding).abs()
            distance_out = F.relu(delta - user_offset_embedding - item_offset_embedding)
            distance_in = torch.min(delta, user_offset_embedding + item_offset_embedding)
            logit = - torch.norm(distance_out, p=1, dim=-1)  -  self.gamma * torch.norm(distance_in, p=1, dim=-1)
            return logit
        elif self.logit_cal == "mul":
            return torch.sum(user_center_embedding * item_center_embedding, dim=-1)
            
        elif self.logit_cal == "distance":
            delta = (item_center_embedding - user_center_embedding).abs()

            dis = - torch.norm(delta, p=1, dim=-1)
            
            return dis
    ## in-batch negative
    def cal_cl_loss(self, cent_embs1, off_embs1, cent_embs2, off_embs2):
        pos_logits = self.cal_logit_box(cent_embs1, off_embs1, cent_embs2, off_embs2)

        all_cent_embs1 = cent_embs1.unsqueeze(1).expand(cent_embs1.shape[0], cent_embs2.shape[0],  self.emb_size)
        all_off_embs1 = off_embs1.unsqueeze(1).expand(cent_embs1.shape[0], cent_embs2.shape[0],  self.emb_size)
        all_cent_embs2 = cent_embs2.unsqueeze(0).expand(cent_embs1.shape[0], cent_embs2.shape[0],  self.emb_size)
        all_offset_embs2 = off_embs2.unsqueeze(0).expand(cent_embs1.shape[0], cent_embs2.shape[0],  self.emb_size)
        
        all_logits = self.cal_logit_box(all_cent_embs1, all_off_embs1, all_cent_embs2, all_offset_embs2)
        
        f = lambda x: torch.exp(x / self.tau)

        between_sim = f(pos_logits)
        all_sim = f(all_logits)

        positive_pairs = between_sim
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.mean(-torch.log(positive_pairs / negative_pairs))
        return loss
    def forward(self, batch):
        user, pos_item, neg_item = batch
        neg_item = torch.stack(neg_item).t().to(self.device)
        user_emb = self.all_embed[:self.n_users, :]
        user_offset = self.all_offset[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_offset = self.all_offset[self.n_users:, :]
        batch_size = user.shape[0]

        neg_item_embs = entity_emb[neg_item.view(-1)]
        neg_item_embs = neg_item_embs.view(batch_size, -1, self.emb_size)
        user_agg_embs, user_agg_offsets, item_agg_embs, item_agg_offset = \
            self.gcn(user_emb, user_offset, entity_emb, entity_offset, self.graph)

        pos_user_embs, pos_user_offsets = user_agg_embs[user], user_agg_offsets[user]
        pos_item_embs, pos_item_offset = item_agg_embs[pos_item], item_agg_offset[pos_item]
        neg_item_embs, neg_item_offset = item_agg_embs[neg_item.view(-1)], item_agg_offset[neg_item.view(-1)]
        
        ### CL Loss
        # u_user_agg_embs, u_user_agg_offsets, u_item_agg_embs, u_item_agg_offset = \
        #     self.gcn(user_emb, user_offset, entity_emb, entity_offset,self.graph_none)
        
        # u_user_cl_loss = self.cal_cl_loss(pos_user_embs, pos_user_offsets, u_user_agg_embs[user], u_user_agg_offsets[user])
        # u_item_cl_loss = self.cal_cl_loss(pos_item_embs, pos_item_offset, u_item_agg_embs[pos_item], u_item_agg_offset[pos_item])
        
        # i_user_agg_embs, i_user_agg_offsets, i_item_agg_embs, i_item_agg_offset = \
        #     self.gcn(user_emb, user_offset, entity_emb, entity_offset,self.relation_embedding, self.relation_offset_embedding,self.graph_i)
        # i_user_cl_loss = self.cal_cl_loss(pos_user_embs, pos_user_offsets, i_user_agg_embs[user], i_user_agg_offsets[user])
        # i_item_cl_loss = self.cal_cl_loss(pos_item_embs, pos_item_offset, i_item_agg_embs[pos_item], i_item_agg_offset[pos_item])
        cl_loss =  0 #(u_user_cl_loss + u_item_cl_loss ) * 0.1
        ### Main Loss
        pos_scores = self.cal_logit_box(pos_user_embs, pos_user_offsets, pos_item_embs, pos_item_offset)
        neg_scores = self.cal_logit_box(pos_user_embs, pos_user_offsets, neg_item_embs, neg_item_offset)
        
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # pos_vol = self.cal_logit_vol(pos_user_embs, pos_user_offsets, pos_item_embs, pos_item_offset)
        # neg_vol = self.cal_logit_vol(pos_user_embs, pos_user_offsets, neg_item_embs, neg_item_offset)

        # vol_loss = -1 * torch.mean(nn.LogSigmoid()(pos_vol - neg_vol))

        # neg_scores = F.logsigmoid(-neg_scores)
        # pos_scores = F.logsigmoid(pos_scores )
        # positive_sample_loss = - pos_scores.mean()
        # negative_sample_loss = - neg_scores.mean()
        # mf_loss = (positive_sample_loss + negative_sample_loss)

        ### CL
        

        regularizer = ( torch.norm(user_emb[user]) ** 2
                       + torch.norm(entity_emb[pos_item]) ** 2
                       + torch.norm(entity_emb[neg_item.reshape(-1)]) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # vol_loss = self.vol_loss(pos_user_offsets) + self.vol_loss(pos_item_offset) + self.vol_loss(neg_item_offset)
        
        return mf_loss + emb_loss + cl_loss , mf_loss, emb_loss
    def vol_loss(self, offset_emb, margin=0.7):
        return offset_emb.norm(dim=-1).mean()

    def generate(self, users):
        user_emb, entity_emb = torch.split(self.all_embed, [self.n_users, self.n_entities])
        user_offset, entity_offset = torch.split(self.all_offset, [self.n_users, self.n_entities])
        user_agg_embs, user_agg_offset, item_agg_embs, item_agg_offset = \
            self.gcn(user_emb, user_offset, entity_emb, entity_offset, self.graph)
        
        user_embs = torch.cat([user_agg_embs, user_agg_offset], axis=-1)
        item_embs = torch.cat([item_agg_embs, item_agg_offset], axis=-1)
        # user_embs = user_embs[:, 0, :]
        # item_embs = item_embs[:, 0, :]
        return user_embs, item_embs


    def rating(self, user_embs, entity_embs, same_dim=False):
        if same_dim:
            user_agg_embs, user_agg_offset = torch.split(user_embs, [self.emb_size, self.emb_size], dim=-1)
            entity_agg_embs, entity_agg_offset = torch.split(entity_embs, [self.emb_size, self.emb_size], dim=-1)
            return self.cal_logit_box(user_agg_embs, user_agg_offset, entity_agg_embs, entity_agg_offset)
        else:
            n_users = user_embs.shape[0]
            n_entities = entity_embs.shape[0]
            user_embs = user_embs.unsqueeze(1).expand(n_users, n_entities,  self.emb_size * 2)
            user_agg_embs, user_agg_offset = torch.split(user_embs, [self.emb_size, self.emb_size], dim=-1)

            entity_embs = entity_embs.unsqueeze(0).expand(n_users, n_entities,  self.emb_size * 2)
            entity_agg_embs, entity_agg_offset = torch.split(entity_embs, [self.emb_size, self.emb_size], dim=-1)

            return self.cal_logit_box(user_agg_embs, user_agg_offset, entity_agg_embs, entity_agg_offset)
