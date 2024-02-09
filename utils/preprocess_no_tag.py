import pandas as pd
import numpy as np
import os
from collections import defaultdict

datasets = [{
    "name": "amazon-movies-tv",
    "file": "movies_tv.csv"
},{
    "name": "movielens-100k",
    "file": "movielens_100k.csv"
}]
def remap(data_map):
    ks = []
    vs = []
    for k, v in data_map.items():
        ks.append(k)
        vs.append(v)
    
    return {"org_id": ks, "remap_id": vs}
for dataset in datasets:
    target_data_path = f"../data/{dataset['name']}"
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)

    target_train_file = os.path.join(target_data_path, "train.txt")
    target_valid_file = os.path.join(target_data_path, "val.txt")
    target_test_file = os.path.join(target_data_path, "test.txt")

    target_user_remap_file = os.path.join(target_data_path, "users.txt")
    target_item_remap_file = os.path.join(target_data_path, "items.txt")


    src_data_path= f"../../Cbox4cr/datasets/{dataset['name']}" 
    src_data_file = os.path.join(src_data_path,dataset['file'])
    # item_info = pd.read_csv(src_item_file, sep="\t")
    ## read src data
    data = pd.read_csv(src_data_file)

    data = data[["userID", "itemID", "rating", "timestamp"]]

    ## userID and itemID is scattered, we ought to remap it.
    user_map = {userID: idx for idx, userID in enumerate(data["userID"].unique())}
    item_map = {itemID: idx for idx, itemID in enumerate(data["itemID"].unique())}

    data["userID"] = data["userID"].map(user_map)
    data["itemID"] = data["itemID"].map(item_map)

    pd.DataFrame.from_dict(remap(user_map)).to_csv(target_user_remap_file, sep="\t", index=False)
    pd.DataFrame.from_dict(remap(item_map)).to_csv(target_item_remap_file, sep="\t", index=False)

    data = data.sort_values("timestamp").reset_index(drop=True)
    # user_counts = data.value_counts("userID")
    user2item = data[["userID", "itemID"]].drop_duplicates().reset_index(drop=True)

    ####leave one out 
    ## consider the last item that each users interacted with as the test data
    test_data = data[~data[["userID"]].duplicated(keep='last')].reset_index(drop=True)
    test_data = test_data[test_data["rating"] > 3]
    train_valid_data = data[data[["userID"]].duplicated(keep='last')].reset_index(drop=True)
    
    valid_data = train_valid_data[~train_valid_data[["userID"]].duplicated(keep='last')].reset_index(drop=True)
    valid_data = valid_data[valid_data["rating"] > 3]
    train_data = train_valid_data[train_valid_data[["userID"]].duplicated(keep='last')].reset_index(drop=True)
    
    # train_data = train_data[train_data["rating"] > 3]
    def save_uipair(data, file):
        data.to_csv(file, sep="\t",  index=False)
    def save_user(data, file):
        user2item_dict = defaultdict(list)
        data["itemID"] = data["itemID"].astype(str)
        for _, row in data.iterrows():
            user2item_dict[row["userID"]].append(row["itemID"])
        with open(file, "w") as f:
            for user, item_list in user2item_dict.items():
                itemstr = " ".join(item_list)
                f.write(f"{user} {itemstr}\n")

    save_uipair(train_data, target_train_file)
    save_uipair(valid_data, target_valid_file)
    save_uipair(test_data, target_test_file)
    # save_user(train_data, target_train_file)
    # save_user(valid_data, target_valid_file)
    # save_user(test_data, target_test_file)

    # user2tag.to_csv(target_user2tag_file, sep="\t", index=False)
    # item2tag.to_csv(target_item2tag_file, sep="\t", index=False)