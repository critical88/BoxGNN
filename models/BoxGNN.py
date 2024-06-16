'''
Created on January 1, 2024
PyTorch Implementation of BoxGNN
@author: FakeLin (linfake@mail.ustc.edu.cn)
'''
__author__ = "linfake"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch_scatter import scatter_sum, scatter_mean, scatter_min,scatter_max
from torch_scatter.composite import scatter_softmax
import math


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, idx, dim_size):
        ### equation (3) and (5)
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        layer2_act = self.layer2(layer1_act)
        attention = scatter_softmax(src=layer2_act, index=idx, dim=0)
        user_embedding = scatter_sum(attention * embeddings, index=idx, dim=0, dim_size=dim_size)
        return user_embedding

class BoxOffsetIntersection(nn.Module):
    """
    only use max/min to obtain high-order offset embeddings.
    """
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()

    def forward(self, embeddings, idx, dim_size, union=False ):
        ### equation (4) and (6)
        if  union:
            offset = scatter_max(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]
        else:
            offset = scatter_min(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]
        return offset 

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, emb_size, n_hops, n_users, n_items, n_tags, device,
                user2tag, item2tag,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.n_layers = n_hops
        self.emb_size = emb_size
        self.n_users = n_users
        self.n_tags = n_tags
        self.n_items = n_items
        self.n_nodes = self.n_users + self.n_items + self.n_tags
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

        idx =torch.arange(self.n_items + self.n_tags + self.n_users).to(self.device)
        self.all_union_idx = torch.cat([idx, idx], dim=0)


        self.center_net = CenterIntersection(self.emb_size)
        self.offset_net = BoxOffsetIntersection(self.emb_size)
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

    def union(self, embs_list, offset_list, index):
        embs = torch.cat(embs_list, dim=0)
        offset = torch.cat(offset_list, dim=0)
        agg_emb = self.center_net(embs, index, embs_list[0].shape[0])
        ent_offset_emb = F.relu(self.offset_net(offset, index, embs_list[0].shape[0], union=True))
        
        return agg_emb, ent_offset_emb


    def forward(self, user_emb, user_offset_emb, item_emb, item_offset_emb, tag_emb, tag_offset, graph, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            graph = self._sparse_dropout(graph, self.node_dropout_rate)
        _indices = graph._indices()
        head, tail = _indices[0,:], _indices[1, :]

        all_embs = torch.cat([user_emb, item_emb, tag_emb], dim=0)
        all_offset_emb = F.relu(torch.cat([user_offset_emb, item_offset_emb, tag_offset], dim=0))
        

        agg_layer_emb = [all_embs]
        agg_layer_offset = [all_offset_emb]


        for _ in range(self.n_layers):
            history_embs = all_embs[tail]
            ## all logical operation use the same controid aggregation
            agg_emb = self.center_net(history_embs, head, self.n_nodes)

            ## user offset
            ### user-item, intersect
            inter_user_ordinal = (head < self.n_users) & (tail >= self.n_users) & (tail < self.n_users + self.n_items)
            
            inter_user_history_offset = F.relu(all_offset_emb[tail[inter_user_ordinal]])
            inter_user_offset_emb = self.offset_net(inter_user_history_offset, head[inter_user_ordinal], self.n_nodes)
            inter_user_offset_emb = inter_user_offset_emb[:self.n_users]
            
            ### user-tag, union
            user_tag_ordinal =  (head < self.n_users) & (tail >= self.n_users + self.n_items)

            user_tag_history_offset = F.relu(all_offset_emb[tail[user_tag_ordinal]])
            ut_user_offset_emb = self.offset_net(user_tag_history_offset, head[user_tag_ordinal], self.n_nodes, union=True)
            ut_user_offset_emb = ut_user_offset_emb[:self.n_users]

            ### union two part 
            user_offset = torch.cat([inter_user_offset_emb, ut_user_offset_emb], dim=0)
            user_offset = F.relu(self.offset_net(user_offset, self.user_union_idx, inter_user_offset_emb.shape[0]))
            
            ### item offset
            ### intersect all neighboring nodes of item
            item_ordinal = (head >= self.n_users) & (head < self.n_users + self.n_items)

            item_history_offset = F.relu(all_offset_emb[tail[item_ordinal]])
            item_offset = self.offset_net(item_history_offset, head[item_ordinal], self.n_nodes, union=True)
            item_offset = item_offset[self.n_users:self.n_users + self.n_items]
            
            ### tag offset
            ### union all neighboring nodes of tag
            tag_ordinal = (head >= self.n_users + self.n_items) 

            tag_history_offset = F.relu(all_offset_emb[tail[tag_ordinal]])
            tag_offset = self.offset_net(tag_history_offset, head[tag_ordinal], self.n_nodes)
            tag_offset = tag_offset[self.n_users + self.n_items:]

            agg_emb = F.normalize(agg_emb)

            agg_offset_emb = F.relu(torch.cat([user_offset, item_offset, tag_offset], dim=0))
            # agg_offset_emb = torch.cat([agg_offset_emb, all_offset_emb], dim=0)
            # agg_offset_emb = F.relu(self.offset_net(agg_offset_emb, self.all_union_idx, all_offset_emb.shape[0], union=True))
            
            
            agg_layer_emb.append(agg_emb)
            agg_layer_offset.append(agg_offset_emb)
            
            all_embs = agg_emb
            all_offset_emb = agg_offset_emb

        # agg_final_emb, agg_final_offset = self.union(agg_layer_emb, agg_layer_offset, self.all_union_idx)
        agg_final_emb = agg_layer_emb[-1] #torch.stack(agg_layer_emb).mean(dim=0)
        agg_final_offset = agg_layer_offset[-1]#torch.stack(agg_layer_offset).mean(dim=0)

        user_final_emb, item_final_emb, tag_final_emb = torch.split(agg_final_emb, [self.n_users, self.n_items, self.n_tags])

        user_final_offset, item_final_offset, tag_final_offset = torch.split(agg_final_offset, [self.n_users, self.n_items, self.n_tags])


        return user_final_emb, user_final_offset, item_final_emb, item_final_offset,tag_final_emb,tag_final_offset


class Recommender(nn.Module):
    def __init__(self, args, data_stat, adj_mat, user2tag, item2tag):
        super(Recommender, self).__init__()

        self.beta = args.beta
        self.n_users = data_stat['n_users']
        self.n_items = data_stat['n_items']
        self.n_tags = data_stat["n_tags"]
        self.n_nodes = data_stat['n_nodes']  # n_users + n_items + n_tags
        self.n_entities = data_stat['n_items'] + data_stat['n_tags']
        self.decay = args.l2
        self.emb_size = args.dim
        self.n_layers = args.context_hops
        self.logit_cal = args.logit_cal
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda \
                                                                      else torch.device("cpu")

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
                         n_tags=self.n_tags,
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
        
        self.graph = self.add_residual(self.graph)

    def add_residual(self, graph):
        residual_node = torch.arange(self.n_nodes).to(self.device)
        row, col = graph._indices()
        row = torch.cat([row, residual_node], dim=0)
        col = torch.cat([col, residual_node], dim=0)
        val = torch.cat([graph._values(), torch.ones_like(residual_node)])

        return torch.sparse.FloatTensor(torch.stack([row, col]), val, graph.shape).to(self.device)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    
    def cal_logit_box(self, user_center_embedding, user_offset_embedding, item_center_embedding, item_offset_embedding, training=True):
        if self.logit_cal == "box":
            gumbel_beta = self.beta
            t1z, t1Z = user_center_embedding - user_offset_embedding, user_center_embedding + user_offset_embedding
            t2z, t2Z = item_center_embedding - item_offset_embedding, item_center_embedding + item_offset_embedding
            z = gumbel_beta * torch.logaddexp(
                    t1z / gumbel_beta, t2z / gumbel_beta
                )
            z = torch.max(z, torch.max(t1z, t2z))
            # z =  torch.max(t1z, t2z)
            Z = -gumbel_beta * torch.logaddexp(
                -t1Z / gumbel_beta, -t2Z / gumbel_beta
            )
            Z = torch.min(Z, torch.min(t1Z, t2Z))
            # Z = torch.min(t1Z, t2Z)
            euler_gamma = 0.57721566490153286060
            return torch.sum(
                torch.log(
                    F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=1/gumbel_beta) + 1e-23
                ),
                dim=-1,
            )
        else:
            t1z, t1Z = user_center_embedding - user_offset_embedding, user_center_embedding + user_offset_embedding
            t2z, t2Z = item_center_embedding - item_offset_embedding, item_center_embedding + item_offset_embedding
            z = torch.max(t1z, t2z)
            Z = torch.min(t1Z, t2Z)
            return torch.sum(F.relu(Z-z), dim=-1)

    def forward(self, batch):
        user, pos_item, neg_item = batch
        neg_item = torch.stack(neg_item).t().to(self.device)
        user_emb, item_emb, tag_emb = torch.split(self.all_embed, [self.n_users, self.n_items, self.n_tags])
        user_offset, item_offset, tag_offset = torch.split(self.all_offset, [self.n_users, self.n_items, self.n_tags])
        batch_size = user.shape[0]

        ### use logical operation to simulate the GNN.
        user_agg_embs, user_agg_offsets, item_agg_embs, item_agg_offset, tag_agg_emb, tag_agg_offset = \
            self.gcn(user_emb, user_offset, item_emb, item_offset, tag_emb, tag_offset, self.graph)

        pos_user_embs, pos_user_offsets = user_agg_embs[user], user_agg_offsets[user]
        pos_item_embs, pos_item_offset = item_agg_embs[pos_item], item_agg_offset[pos_item]
        neg_item_embs, neg_item_offset = item_agg_embs[neg_item.view(-1)], item_agg_offset[neg_item.view(-1)]

        ### use gumbel-distribution to calculate volume of the intersection box between user and pos/neg item embs.
        pos_scores = self.cal_logit_box(pos_user_embs, pos_user_offsets, pos_item_embs, pos_item_offset)
        neg_scores = self.cal_logit_box(pos_user_embs, pos_user_offsets, neg_item_embs, neg_item_offset)
        
        ### BPR
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        
        ## regularization
        regularizer = ( torch.norm(user_emb[user]) ** 2
                       + torch.norm(item_emb[pos_item]) ** 2
                       + torch.norm(item_emb[neg_item.reshape(-1)]) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        
        return mf_loss + emb_loss  , mf_loss, emb_loss

    def generate(self):
        user_emb, item_emb, tag_emb = torch.split(self.all_embed, [self.n_users, self.n_items, self.n_tags])
        user_offset, item_offset, tag_offset= torch.split(self.all_offset, [self.n_users, self.n_items, self.n_tags])
        user_agg_embs, user_agg_offset, item_agg_embs, item_agg_offset, tag_agg_embs, tag_agg_offset = \
            self.gcn(user_emb, user_offset, item_emb, item_offset, tag_emb, tag_offset, self.graph)
        
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
