
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from .utils_guard import gcn_norm
from torch_sparse import sum as sparsesum
from torch_scatter import scatter_add
class EGCNGuard(nn.Module):
    """
    Efficient GCNGuard

    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, attention_drop=True, threshold=0.1):
        super(EGCNGuard, self).__init__()

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
        # the definition of p0 is confusing comparing the paper and the issue
        # self.p0 = p0
        # https://github.com/mims-harvard/GNNGuard/issues/4
        self.gate = 0. #Parameter(torch.rand(1)) 
        self.prune_edge = True
        self.threshold = threshold

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        

    def forward(self, x, adj):
        if self.layer_norm_first:
            x = self.lns[0](x)
        new_adj = adj
        for i, conv in enumerate(self.convs[:-1]):
            new_adj = self.att_coef(x, new_adj)
            x = conv(x, new_adj)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        new_adj = self.att_coef(x, new_adj)
        x = conv(x, new_adj)
        return x.log_softmax(dim=-1)

    def att_coef(self, features, adj):
        with torch.no_grad():
            row, col = adj.coo()[:2]
            row = row.to(torch.long)  # ← 新增
            col = col.to(torch.long)  # ← 新增
            if row.device != features.device:  # ← 可选的设备对齐
                row = row.to(features.device)
                col = col.to(features.device)

            n_total = features.size(0)
            if features.size(1) > 512 or row.size(0) > 5e5:
                batch_size = int(1e8 // features.size(1))
                bepoch = row.size(0) // batch_size + (row.size(0) % batch_size > 0)
                sims = []
                for i in range(bepoch):
                    st = i * batch_size
                    ed = min((i + 1) * batch_size, row.size(0))
                    sims.append(F.cosine_similarity(features[row[st:ed]], features[col[st:ed]]))
                sims = torch.cat(sims, dim=0)
            else:
                sims = F.cosine_similarity(features[row], features[col])

            mask = torch.logical_or(sims >= self.threshold, row == col)
            row = row[mask]
            col = col[mask]
            sims = sims[mask]

            has_self_loop = (row == col).sum().item()
            if has_self_loop:
                sims[row == col] = 0

            # normalize sims
            deg = scatter_add(sims, row, dim=0, dim_size=n_total)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            sims = deg_inv_sqrt[row] * sims * deg_inv_sqrt[col]

            # add self-loops
            deg_new = scatter_add(torch.ones_like(sims), col, dim=0, dim_size=n_total) + 1
            deg_inv_sqrt_new = deg_new.float().pow_(-1.0)
            # ⚠ 这里原代码用的是 deg_inv_sqrt，容易写错，建议改成如下这一行：
            deg_inv_sqrt_new.masked_fill_(deg_inv_sqrt_new == float('inf'), 0)

            if has_self_loop == 0:
                new_idx = torch.arange(n_total, device=row.device)
                row = torch.cat((row, new_idx), dim=0)
                col = torch.cat((col, new_idx), dim=0)
                sims = torch.cat((sims, deg_inv_sqrt_new), dim=0)
            elif has_self_loop < n_total:
                #print(f"add {n_total - has_self_loop} remaining self-loops")
                new_idx = torch.ones(n_total, device=row.device).bool()
                new_idx[row[row == col]] = False
                new_idx = torch.nonzero(new_idx, as_tuple=True)[0]
                row = torch.cat((row, new_idx), dim=0)
                col = torch.cat((col, new_idx), dim=0)
                sims = torch.cat((sims, deg_inv_sqrt_new[new_idx]), dim=0)
                sims[row == col] = deg_inv_sqrt_new
            else:
                sims[row == col] = deg_inv_sqrt_new

            sims = sims.exp()
            graph_size = torch.Size((n_total, n_total))
            new_adj = SparseTensor(row=row, col=col, value=sims, sparse_sizes=graph_size)
        return new_adj