# BoxGNN

### Tag Collaborative Graph
<p align=center>
  <img src="/r/BoxGNN-607F/TCG.png" width=600>
</p>

Due to time constraints, we could only sketch a rough draft. If the paper is accepted, we will refine this figure. From the figure, it is evident that everyone has a variety of preferences, such as Portable, Digital, Apple, etc., leading to the purchase of different products. At the same time, people may buy the same product for different reasons. If we directly use point embeddings to characterize the various nodes, we cannot adequately express the diversity and uncertainty of users and products.

On the other hand, traditional GNN point aggregation does not seem quite suitable for Tag-aware Recommender System (TRS). For example, considering the "Portable" tag, it is merely an attribute of items and users. However, point aggregation would incorporate too much unnecessary information from its neighbors, which distorts its original meaning. Instead, if we represent it as a box, the "Portable" tag can be understood as a common area shared by iPads, bags, and female users, and the representation of "Portable" tag can be precisely extracted.


### Distribution of Embeddings
<p align=center>
  <img src="/r/BoxGNN-607F/lfgcf_before_2d.jpg" width=400>
  <img src="/r/BoxGNN-607F/lfgcf_after_2d.jpg" width=400>
</p>

<p align=center>
  <img src="/r/BoxGNN-607F/BoxGNN_before_2d.jpg" width=400>
  <img src="/r/BoxGNN-607F/BoxGNN_after_2d.jpg" width=400>
</p>
We visualized the distribution of the centroid points of Box on the Movielens dataset. For comparison, we also visualized the distribution of node embeddings from LFGCF. It's noteworthy that the two images on the left represent the distribution of points before the GNN operation, while the image on the right represents the distribution of points after the GNN operation. From the Figure (a), we can see that before performing GNN, nodes of the same type in the LFGCF model are close to each other. This indicates that they have a high similarity among nodes of the same type, lacking differentiation and diversity. Furthermore, from the Figure (b), it is apparent that after performing GNN, the clustering phenomenon becomes more significant.

From the Figure (c), we can observe that before performing GNN operations, the distribution of points in BoxGNN is quite uniform. This indicates that our method possess sufficient diversity and differentiation, and this part of the gain mainly comes from Box modeling, which aligns with our motivation for using Box to model diversity of user interests. After the GNN operation, Figure (d) shows several blue clustering points (items), which is quite reasonable since the purpose of GNN itself is to explicitly aggregate nodes with similar information. Furthermore, the distribution of these clustering points is still quit uniform, rather than forming a few large clustering points like in Figure (b).

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
