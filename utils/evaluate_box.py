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





def get_orginal_kg(model):
    n_relation = (model.n_relations - 1) / 2 - 1
    head, tail = model.edge_index
    edge_type = model.edge_type - 1

    org_index = torch.where(edge_type <= n_relation)[0]

    can_triplets_np = np.array([head[org_index].cpu().numpy(), edge_type[org_index].cpu().numpy(), tail[org_index].cpu().numpy()])

    can_triplets_np = can_triplets_np.T

    return can_triplets_np

def get_masked_info(model, mask):
    n_relation = (model.n_relations - 1) / 2 - 1
    # can_triplets_np = np.loadtxt(f"{args.data_path}{args.dataset}/{args.kg_file}.txt", dtype=np.int32)
    
    # saved_mask_sequence = np.load(f"{args.data_path}{args.dataset}/final_masked_sequence.npy")
    # mask = (saved_mask_sequence < 0.05).all(axis=0)

    
    head, tail = model.edge_index
    edge_type = model.edge_type - 1

    can_triplets_np = get_orginal_kg(model)


    head, tail, edge_type = head.cpu().numpy(), tail.cpu().numpy(), edge_type.cpu().numpy()
    
    # masked_entities = np.unique(tail[mask])
    
    # head_mask = np.where(np.isin(head, masked_entities))[0]
    
    # mask = np.union1d(mask, head_mask)

    removed_triplets = np.array([head[mask], edge_type[mask], tail[mask]]).transpose(1, 0)
    
    # removed_triplets = np.loadtxt(f"{args.data_path}{args.dataset}/final_masked_edges.txt", dtype=np.int32)
    
    masked_triplest_df = pd.DataFrame(removed_triplets, columns=["h", "r", "t"])
    
    all_triplets_df = pd.DataFrame(can_triplets_np, columns=["h", "r", "t"])

    masked_triplest_df["match"] = masked_triplest_df["h"].astype(str).str.cat(masked_triplest_df["r"].astype(str), sep='_').str.cat(masked_triplest_df["t"].astype(str), sep="_")
    
    all_triplets_df["match"] = all_triplets_df["h"].astype(str).str.cat(all_triplets_df["r"].astype(str), sep='_').str.cat(all_triplets_df["t"].astype(str), sep="_")

    return masked_triplest_df, all_triplets_df

def save_unpruned_node(model, mask_file, saved_pruned_kg_file):

    mask = np.load(mask_file)
    masked_triplest_df, all_triplets_df = get_masked_info(model, mask)
    
    intersected_triplets = all_triplets_df[~(all_triplets_df["match"].isin(masked_triplest_df["match"]))]
    intersected_triplets[["h", "r", "t"]].to_csv(saved_pruned_kg_file, index=False, header=None, sep=" ")

def test(model, train_user_set, test_user_set, data_stat):
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
    
    histories = retrieve_all_histories(user_ids, graph, device)
    users = torch.LongTensor(user_ids).to(device)
    query_embs, query_offsets = model.generate(users, histories)
    
    
    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)

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
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach()

        maxK = max(Ks)
        for i, u in enumerate(user_list_batch):
            try:
                training_items = train_user_set[u]
            except Exception:
                training_items = []
            # user u's items in the test set
            rate_batch[i][training_items] = -np.inf
        
        rate_topK_val, rate_topk_idx = torch.topk(rate_batch, k=maxK, largest=True, dim=-1)
        # user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = []
        for u, rate_topk in zip(user_list_batch, rate_topk_idx):
            r = []
            user_pos_test = test_user_set[u]
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
