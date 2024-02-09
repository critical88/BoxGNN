from .metrics import *
from .parser import parse_args

from .metrics1 import calc_metrics_at_k
import torch
import numpy as np
import multiprocessing
import heapq
from time import time
from tqdm import tqdm
import pandas as pd
cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


def retrieve_all_histories(user_ids, n_users, graph, device):
    graph = np.array([graph.row, graph.col])
    idx = np.where(np.isin(graph[0], user_ids))[0]
    histories = graph[:, idx]
    return torch.LongTensor(histories.T).to(device)

def test(model, args, dataset, train_user_set, test_user_set, data_stat):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0., 
              "ratings": []}

    n_items = data_stat['n_items']
    n_users = data_stat['n_users']

    # pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0
    if args.model.__contains__("box"):
        # histories = retrieve_all_histories(test_users, n_users, graph, device)
        users = torch.LongTensor(test_users).to(device)
        embs = model.generate(users,)
    else:
        embs = model.generate()

    user_gcn_emb, entity_gcn_emb = embs
    
    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if args.eval_rnd_neg:
            ### 随机选取100个负样本（已去重和去除训练集内容）
            negatives = dataset.negative_sampling(user_batch, 100)

            neg_items = torch.LongTensor(negatives).to(device)
            pos_items = []
            for u in user_list_batch:
                pos_items.append(test_user_set[u])
            pos_items = torch.LongTensor(pos_items).to(device)

            ### 100个负样本，正样本放在第一个。
            all_items = torch.cat([pos_items, neg_items], dim=-1)
            i_g_embddings = entity_gcn_emb[all_items]
            u_g_embeddings = u_g_embeddings.unsqueeze(1).expand(u_g_embeddings.shape[0], all_items.shape[1], u_g_embeddings.shape[1])
            rate_batch = model.rating(u_g_embeddings, i_g_embddings, same_dim=True).detach()
        else:
            all_items = torch.LongTensor(np.array(range(0, n_items))).to(device)
            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = torch.zeros((len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = all_items.view(n_items, -1)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach()

        
            for i, u in enumerate(user_list_batch):
                try:
                    training_items = train_user_set[u]
                    test_items = test_user_set[u]
                except Exception:
                    training_items = []
                # user u's items in the test set
                item_scores =  rate_batch[i][test_items].clone()
                rate_batch[i][training_items] = -np.inf
                rate_batch[i][test_items] = item_scores

        maxK = max(Ks)
        rate_topK_val, rate_topk_idx = torch.topk(rate_batch, k=maxK, largest=True, dim=-1)
        # user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = []
        for u, rate_topk in zip(user_list_batch, rate_topk_idx):
            r = []
            user_pos_test = test_user_set[u]
            if args.eval_rnd_neg:
                for idx in rate_topk:
                    ### 100个负样本，正样本放在第一个。
                    if idx.item() == 0:
                        r.append(1)
                    else:
                        r.append(0)
            else:
                for idx in rate_topk:
                    if idx.item() in user_pos_test:
                        r.append(1)
                    else:
                        r.append(0)
            performance = get_performance(user_pos_test, r, Ks)

            batch_result.append(performance)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
    assert count == n_test_users
    # pool.close()
    return result
