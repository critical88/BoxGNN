import torch
from utils.parser import parse_args

import numpy as np
import random
from torch.utils.data import DataLoader
import os
from time import time
from prettytable import PrettyTable
import json

from utils.evaluate import test
from utils.helper import early_stopping

# 创建一个具有3种节点类型和3种边类型的异构图

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_args_config(args):
    saved_config_file = os.path.join(args.save_dir, "config.json")
    arg_dict = json.dumps(args.__dict__)
    with open(saved_config_file, "w") as f:
        f.write(arg_dict)

def train(args):
    seed = 2024
    seed_all(seed)
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")


    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    save_args_config(args)

    
    
    if args.model == "dspr":
        from utils.data_dspr import DataUtils,TrainDataset
        dataUtils = DataUtils(args)
        train_data, mdata, user2tag, item2tag, data_dict, data_stat = dataUtils.read_files()
        user_adj, item_adj = dataUtils.build_graph(mdata, user2tag, item2tag, data_stat)
    elif args.model == "lfgcf":
        from utils.data_lfgcf import DataUtils,TrainDataset
        dataUtils = DataUtils(args)
        train_data, val_data, test_data, user2tag, item2tag, data_dict, data_stat = dataUtils.read_files()
        user_adj, item_adj, tag_adj = dataUtils.build_graph(train_data, user2tag, item2tag, data_stat)
    else:
        from utils.data import DataUtils, TrainDataset
        dataUtils = DataUtils(args)
        train_data, val_data, test_data, user2tag, item2tag, data_dict, data_stat = dataUtils.read_files()
        adj_cf_mat, norm_cf_mat, mean_cf_mat = dataUtils.build_interact_matrix(train_data, data_stat)

        adj_mat, norm_mat, mean_mat = dataUtils.build_graph(train_data, user2tag, item2tag, data_stat)
    

    train_data = torch.LongTensor(train_data).to(device)
    
    train_dataset = TrainDataset(args, train_data, data_dict, data_stat)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    

    if args.model =="box_gcn":
        from models.BoxGCN import Recommender
        model = Recommender(args, data_stat, norm_mat, user2tag, item2tag)
    elif args.model =="box_gumbel_gcn":
        from models.BoxGCN_gumbel import Recommender
        model = Recommender(args, data_stat, norm_mat, user2tag, item2tag)
    elif args.model == "lightgcn":
        from models.LightGCN import Recommender
        model = Recommender(args, data_stat, norm_cf_mat)
    elif args.model == "ngcf":
        from models.NGCF import Recommender
        model = Recommender(args, data_stat, norm_cf_mat)
    elif args.model =="bpr":
        from models.BPR import Recommender
        model = Recommender(args, data_stat)
    elif args.model == "dspr":
        from models.DSPR import Recommender
        model = Recommender(args, data_stat, user_adj, item_adj)
    elif args.model == "lfgcf":
        from models.LFGCF import Recommender
        model = Recommender(args, data_stat, user_adj, item_adj, tag_adj)
    else:
        raise Exception(f"unsupported model: {args.model}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    best_ret = None
    best_test_ret = None
    
    best_epoch = 0

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    ### negative sampling
    
    print("start training ... ")
    for epoch in range(args.epoch):
        train_dataset.create_all_negative_sample()
        model.train()
        loss = 0
        train_s_t = time()
        for batch in train_dataloader:
            
            batch_loss, _, _ = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        """training CF"""

        train_e_t = time()
        if (epoch+1) % args.eval_step == 0 or epoch == 0:
            """testing"""
            model.eval()
            ### valid dataset
            test_s_t = time()
            ret = test(model, args, train_dataset, data_dict["user2item"], data_dict["val_user2item"], data_stat)
            # ret = evaluate(model, 2048, n_items, user_dict["train_user_set"], user_dict["test_user_set"], device)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Valid Epoch", "training time", "tesing time", "Loss", "recall", "hit", "ndcg", "precision"]
            # ret = ret[20]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret["hit_ratio"], ret['ndcg'], ret['precision']]
            )
            print(train_res, flush=True)
            
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            #### test dataset
            test_s_t = time()
            test_ret = test(model, args, train_dataset, data_dict["user2item"], data_dict["test_user2item"], data_stat)
            # ret = evaluate(model, 2048, n_items, user_dict["train_user_set"], user_dict["test_user_set"], device)
            test_e_t = time()

            test_res = PrettyTable()
            test_res.field_names = ["Test Epoch", "training time", "tesing time", "Loss",  "recall", "hit", "ndcg", "precision"]
            # ret = ret[20]
            test_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret["hit_ratio"], test_ret['ndcg'], test_ret['precision']]
            )
            print(test_res, flush=True)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0:
                best_epoch = epoch
                best_ret = ret
                best_test_ret = test_res
                # torch.save(model.state_dict(), os.path.join(save_dir, "checkpoint.ckpt"))
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()), flush=True)
    print(best_test_ret,flush=True)
    print('early stopping at %d, valid recall@10:%.4f, ndcg@10:%.4f' % (best_epoch, cur_best_pre_0, best_ret['ndcg'][0]))
    # print("test recall@10:%.4f, ndcg@10:%.4f" % (best_test_ret['recall'][0], best_test_ret['ndcg'][0]))
def predict(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    save_dir = "model_{}_{}".format(args.model,args.dataset)
    save_dir = os.path.join(args.out_dir,save_dir)
    args.save_dir = save_dir
    if not args.pretrain_model_path:
        train(args)
    else:
        predict(args)