import pandas as pd
import numpy as np
import os
from collections import defaultdict

np.random.seed(2022)
datasets = [
{
    "name": "delicious",
    "file": "user_taggedbookmarks-timestamps.dat",
    "user_col": "userID",
    "item_col": "bookmarkID",
    "tag_col": "tagID",
    "item_file": "bookmarks.dat",
    "item_title_col": "title",
    "freq_min": 15
},
{
    "name": "lastfm",
    "file": "user_taggedartists-timestamps.dat",
    "user_col": "userID",
    "item_col": "artistID",
    "tag_col": "tagID",
    "item_file": "artists.dat",
    "item_title_col": "name",
    "freq_min": 5
},{
    "name": "movielens",
    "file": "user_taggedmovies-timestamps.dat",
    "user_col": "userID",
    "item_col": "movieID",
    "tag_col": "tagID",
    "item_file": "movies.dat",
    "item_title_col": "title",
    "freq_min": 5
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
    target_user2tag_file = os.path.join(target_data_path, "user2tag.txt")
    target_item2tag_file = os.path.join(target_data_path, "item2tag.txt")

    target_user_remap_file = os.path.join(target_data_path, "users.txt")
    target_item_remap_file = os.path.join(target_data_path, "items.txt")
    target_tag_remap_file = os.path.join(target_data_path, "tags.txt")


    src_data_path= f"../../raw_data/{dataset['name']}" 
    src_data_file = os.path.join(src_data_path,dataset['file'])
    src_item_file = os.path.join(src_data_path, dataset['item_file'])
    # item_info = pd.read_csv(src_item_file, sep="\t")
    ## read src data
    data = pd.read_csv(src_data_file, sep="\t")

    data["userID"] = data[dataset["user_col"]]
    data["itemID"] = data[dataset["item_col"]]
    data["tagID"] = data[dataset["tag_col"]]
    data = data[["userID", "itemID", "tagID", "timestamp"]]

    ## userID and itemID is scattered, we ought to remap it.
    

    tag_counts = data.value_counts("tagID")
    preserved_tags = tag_counts[tag_counts >= dataset['freq_min']].index
    data = data[data["tagID"].isin(preserved_tags)]
    
    user_map = {userID: idx for idx, userID in enumerate(data["userID"].unique())}
    item_map = {itemID: idx for idx, itemID in enumerate(data["itemID"].unique())}
    tag_map = {tagID: idx for idx, tagID in enumerate(data["tagID"].unique())}
    data["userID"] = data["userID"].map(user_map)
    data["itemID"] = data["itemID"].map(item_map)
    data["tagID"] = data["tagID"].map(tag_map)

    pd.DataFrame.from_dict(remap(user_map)).to_csv(target_user_remap_file, sep="\t", index=False)
    pd.DataFrame.from_dict(remap(item_map)).to_csv(target_item_remap_file, sep="\t", index=False)
    pd.DataFrame.from_dict(remap(tag_map)).to_csv(target_tag_remap_file, sep="\t", index=False)
    
    data["user_item"] = data["userID"].astype(str) + "_" + data["itemID"].astype(str)
    user2item = data[["userID", "itemID"]].drop_duplicates().reset_index(drop=True)


    ## split the data into train(80%), val(10%), test(10%) 
    train_data = user2item.sample(frac=0.8)
    train_data_user_item = train_data["userID"].astype(str) + "_" + train_data["itemID"].astype(str)
    
    # train_data = train_data[["userID", "itemID"]]
    left_data = user2item[~(user2item.index.isin(train_data.index))].reset_index(drop=True)
    left_data = left_data[["userID", "itemID"]].drop_duplicates().reset_index(drop=True)
    valid_data = left_data.sample(frac=0.5)
    test_data = left_data[~(left_data.index.isin(valid_data.index))]
    ### only select the tags involved in the training data
    
    train_data = data[data["user_item"].isin(train_data_user_item)][["userID", "itemID", "tagID"]]
    
    train_data_tag = data[data["user_item"].isin(train_data_user_item)]
    user2tag = train_data_tag[["userID", "tagID"]].drop_duplicates()
    item2tag = train_data_tag[["itemID", "tagID"]].drop_duplicates()

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

    user2tag.to_csv(target_user2tag_file, sep="\t", index=False)
    item2tag.to_csv(target_item2tag_file, sep="\t", index=False)