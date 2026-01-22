
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from .utils_guard import gcn_norm
import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class GCNGuard(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, attention_drop=True):
        super(GCNGuard, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        # specific designs from GNNGuard
        self.attention_drop = attention_drop
        # the definition of p0 is confusing regarding the paper and the issue
        # self.p0 = p0
        # https://github.com/mims-harvard/GNNGuard/issues/4
        self.gate = 0. #Parameter(torch.rand(1)) 
        if self.attention_drop:
            self.drop_learn = torch.nn.Linear(2, 1)
        self.prune_edge = True

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.attention_drop:
            self.drop_learn.reset_parameters()
        # self.gate.weight = 0.0

    def forward(self, x, adj):
        if self.layer_norm_first:
            x = self.lns[0](x)
        adj_memory = None
        for i, conv in enumerate(self.convs[:-1]):
            # print(f"{i} {sum(sum(torch.isnan(x)))} {x.mean()}")
            if self.prune_edge:
                # old_edge_size = adj.coo()[0].size(0)
                new_adj = self.att_coef(x, adj)
                if adj_memory != None and self.gate > 0:
                    # adj_memory makes the performance even worse
                    adj_memory = self.gate * adj_memory.to_dense() + (1 - self.gate) * new_adj.to_dense()
                    row, col = adj_memory.nonzero()[:2]
                    adj_values = adj_memory[row,col]
                else:
                    adj_memory = new_adj
                    row, col, adj_values = adj_memory.coo()[:3]
                # adj_values[torch.isnan(adj_values)] = 0.0
                edge_index = torch.stack((row, col), dim=0)
                # print(f"{sum(torch.isnan(adj_values))} {adj_values.mean()}")
                # adj_values = adj_memory[row, col]
                # print(edge_index,adj_values)
                # print(f"Pruned edges: {i} {old_edge_size-adj.coo()[0].size(0)}")
                adj = new_adj
            x = conv(x, edge_index, edge_weight=adj_values)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.prune_edge:
            # old_edge_size = adj.coo()[0].size(0)
            new_adj = self.att_coef(x, adj)
            if adj_memory != None and self.gate > 0:
                # adj_memory makes the performance even worse
                adj_memory = self.gate * adj_memory.to_dense() + (1 - self.gate) * new_adj.to_dense()
                row, col = adj_memory.nonzero()[:2]
                adj_values = adj_memory[row,col]
            else:
                adj_memory = new_adj
                row, col, adj_values = adj_memory.coo()[:3]
            # adj_values[torch.isnan(adj_values)] = 0.0
            edge_index = torch.stack((row, col), dim=0)
            # print(f"{sum(torch.isnan(adj_values))} {adj_values.mean()}")
            # adj_values = adj_memory[row, col]
            # print(edge_index,adj_values)
            # print(f"Pruned edges: {i} {old_edge_size-adj.coo()[0].size(0)}")
            adj = new_adj
        x = conv(x, edge_index, edge_weight=adj_values)
        # x = self.convs[-1](x, adj)
        # exit()
        return x.log_softmax(dim=-1)


    def att_coef(self, features, adj):
        
        edge_index = adj.coo()[:2]

        n_node = features.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.attention_drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges
        # print("att", att_dense_norm[0,0])
        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            # print(lam.shape)
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        # att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).to(features.device)
        # att_adj = torch.tensor(att_adj, dtype=torch.int64).to(features.device)
        shape = (n_node, n_node)
        new_adj = SparseTensor(row=torch.LongTensor(row).to(features.device), 
                            col=torch.LongTensor(col).to(features.device), 
                            value=att_edge_weight, sparse_sizes=torch.Size(shape))
        
        # new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        # print(new_adj)
        return new_adj
