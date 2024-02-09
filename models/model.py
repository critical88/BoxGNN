import torch
import torch.nn


class BoxELayer(nn.Module):
    def __init__(
        self, meta_path_patterns, in_size, out_size, layer_num_heads, dropout
    ):
        super(BoxELayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_path_patterns = list(
            tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns
        )

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        # obtain metapath reachable graph
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path_pattern in self.meta_path_patterns:
                self._cached_coalesced_graph[
                    meta_path_pattern
                ] = dgl.metapath_reachable_graph(g, meta_path_pattern)

        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, in_size, out_size, dropout=0.1):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Transform weights for different types of edges
        self.W_T = nn.ModuleDict(
            {
                name: nn.Linear(in_size, out_size, bias=False)
                for name in g.etypes
            }
        )

        # Attention weights for different types of edges
        self.W_A = nn.ModuleDict(
            {name: nn.Linear(out_size, 1, bias=False) for name in g.etypes}
        )

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data["h"] = feat_dict[
                dsttype
            ]  # nodes' original feature
            g.nodes[srctype].data["h"] = feat_dict[srctype]
            g.nodes[srctype].data["t_h"] = self.W_T[etype](
                feat_dict[srctype]
            )  # src nodes' transformed feature

            # compute the attention numerator (exp)
            g.apply_edges(fn.u_mul_v("t_h", "h", "x"), etype=etype)
            g.edges[etype].data["x"] = torch.exp(
                self.W_A[etype](g.edges[etype].data["x"])
            )

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e("x", "m"), fn.sum("m", "att"))
        g.multi_update_all(funcs, "sum")

        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(
                fn.e_div_v("x", "att", "att"), etype=etype
            )  # compute attention weights (numerator/denominator)
            funcs[etype] = (
                fn.u_mul_e("h", "att", "m"),
                fn.sum("m", "h"),
            )  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, "sum")

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.relu_(g.nodes[ntype].data["h"]))
            )  # apply activation, layernorm, and dropout

        return feat_dict


class BoxE(nn.Module):
    def __init__(
        self, args, n_users, n_items, n_tags, graph
    ):
        super(BoxE, self).__init__()
        self.n_nodes = n_users + n_items + n_tags
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags

        self.l2 = args.l2
        self.emb_size = args.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        
        self.device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.graph = graph

        # embeddings for different types of nodes, h0
        initializer = nn.init.xavier_uniform_
        self.embed_dict = nn.ParameterDict(
            {
                ntype: nn.Parameter(
                    initializer(torch.empty(g.num_nodes(ntype), self.emb_size))
                )
                for ntype in g.ntypes
            }
        )

        # relational neighbor aggregation, this produces h1
        self.RelationalAGG = RelationalAGG(g, in_size, out_size)

        
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict(
            {
                key: BoxELayer(value, in_size, out_size, num_heads, dropout)
                for key, value in self.meta_path_patterns.items()
            }
        )

        # layers to combine h0, h1, and h2
        # used to update node embeddings

        # network to score the node pairs
        # self.pred = nn.Linear(out_size, out_size)
        self.dropout = nn.Dropout(mess_dropout_rate)
        # self.fc = nn.Linear(out_size, 1)

    def forward(self, g, user_key, item_key, user_idx, item_idx):
        # relational neighbor aggregation, h1
        h1 = self.RelationalAGG(g, self.feature_dict)

        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](g, self.feature_dict[key])

        # update node embeddings
        user_emb = torch.cat((h1[user_key], h2[user_key]), 1)
        item_emb = torch.cat((h1[item_key], h2[item_key]), 1)
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        user_emb = self.user_layer2(
            torch.cat((user_emb, self.feature_dict[user_key]), 1)
        )
        item_emb = self.item_layer2(
            torch.cat((item_emb, self.feature_dict[item_key]), 1)
        )

        # Relu
        user_emb = F.relu_(user_emb)
        item_emb = F.relu_(item_emb)

        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)

        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        interaction = user_feat * item_feat

        # score the node pairs
        pred = self.pred(interaction)
        pred = self.dropout(pred)  # dropout
        pred = self.fc(pred)
        pred = torch.sigmoid(pred)

        return pred.squeeze(1)

class BoxE(nn.Module):
    def __init__(self, args, n_users, n_items, n_tags, graph):
        super(BoxE, self).__init__()

        self.n_nodes = n_users + n_items + n_tags
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags

        self.l2 = args.l2
        self.emb_size = args.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        
        self.device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.graph = graph

        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)

