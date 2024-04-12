# BoxGNN

### Tag Collaborative Graph


### Distribution of Embeddings
<p align=center>
  <img src="/lfgcf_after_2d.jpg" width=400>
  <img src="/BoxGNN_after_2d.jpg" width=400>
</p>

We visualized the distribution of the centroid points of Box on the Movielens dataset. For comparison, we also visualized the distribution of node embeddings from LFGCF. Note that these distributions are post-GNN aggregation. From the left-hand figure, we can see that the distribution for LFGCF mainly converges around certain points, indicating that their learned embeddings have a high degree of similarity, leading to a lack of differentiation and diversity. From the right-hand figure, we can see that the distribution of centroid points of BoxGNN is more uniform, especially for user (red) nodes, who do not cluster around specific points and are surrounded by a variety of tags and items. This indicates sufficient differentiation and diversity among users, which aligns with our motivation.

### Run the Experiments
MovieLens
```shell
python main.py --dataset movielens --model box_gumbel_gcn --gumbel_beta 0.2
```
LastFm
```shell
python main.py --dataset lastfm --model box_gumbel_gcn --gumbel_beta 0.3
```

E-shop
```shell
python main.py --dataset e-shop --model box_gumbel_gcn --gumbel_beta 0.2
```

`E-shop` dataset will be coming soon.
