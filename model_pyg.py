# model_pyg.py  —— 仅展示与本次需求相关的四个组件：RobustGCN / GNNGuard / RGAT / FLAG
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, sum as sparsesum, mul, set_diag
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import copy
import dgl
# ---------- 通用的GCN归一化（SparseTensor版），与您上传代码一致 ----------
def gcn_norm(adj_t: SparseTensor, order: float = -0.5, add_self_loops: bool = True) -> SparseTensor:
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

# ===============================================================
# RobustGCN（与您上传的实现一致，默认 dropout=0.5，ELU/ ReLU 激活，方差采样）
# ===============================================================
class RobustGCNConv(nn.Module):
    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
        super().__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv  = nn.Linear(in_features, out_features)
        self.act0, self.act1 = act0, act1
        self.initial = initial
        self.dropout = dropout

    def reset_parameters(self):
        self.mean_conv.reset_parameters()
        self.var_conv.reset_parameters()

    def forward(self, mean: Tensor, var: Optional[Tensor] = None,
                adj0: Optional[SparseTensor] = None, adj1: Optional[SparseTensor] = None):
        if self.initial:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var  = mean
            mean = self.act0(self.mean_conv(mean))
            var  = self.act1(self.var_conv(var))
        else:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var  = F.dropout(var,  p=self.dropout, training=self.training)
            mean = self.act0(self.mean_conv(mean))
            var  = self.act1(self.var_conv(var)) + 1e-6
            att  = torch.exp(-var)
            mean = mean * att
            var  = var  * att * att
            mean = adj0 @ mean
            var  = adj1 @ var
        return mean, var

class APPNP(torch.nn.Module):
    # code copied from dgl examples
    def __init__(self, in_feats, h_feats, num_classes):
        super(APPNP, self).__init__()
        self.mlp = torch.nn.Linear(in_feats, num_classes)
        self.conv = dgl.nn.APPNPConv(k=3, alpha=0.5)

    def forward(self, g, in_feat):
        in_feat = self.mlp(in_feat)
        h = self.conv(g, in_feat)
        return h

    def train(self, g, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        test_idx = None
        best_model = None
        for e in range(200):
            # Forward
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_model = copy.deepcopy(model)
                best_val_acc = val_acc
                best_test_acc = test_acc
                test_idx = test_mask.nonzero(as_tuple=True)[0][(pred[test_mask] == labels[test_mask]).nonzero(as_tuple=True)[0]]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Victim model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(best_val_acc.item()*100, best_test_acc.item()*100))
        return best_model, test_idx

class RobustGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(in_channels, hidden_channels, initial=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(RobustGCNConv(hidden_channels, hidden_channels, dropout=dropout))
        self.layers.append(RobustGCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, x: Tensor, adj: SparseTensor):
        adj0, adj1 = gcn_norm(adj), gcn_norm(adj, order=-1.0)
        mean, var = x, x
        for conv in self.layers:
            mean, var = conv(mean, var=var, adj0=adj0, adj1=adj1)
        sample = torch.randn_like(var)
        out = mean + sample * torch.pow(var, 0.5)
        return out.log_softmax(dim=-1)
# 参考实现与缺省设置来源：您上传的 model_pyg.py（RobustGCN / RobustGCNConv / gcn_norm）。:contentReference[oaicite:2]{index=2}

# ===============================================================
# GNNGuard（GCNGuard 实现，按特征余弦相似度对边进行权重/裁剪；与上传一致）
# ===============================================================
# from typing import Optional
# import torch as th
# import dgl
# from dgl.nn import APPNPConv
# from .base import VictimBase
#
#
# class APPNP(VictimBase):
#     def __init__(self, in_dim:int, hid:int, out_dim:int, k:int=10, alpha:float=0.1, dropout:float=0.5):
#         super().__init__()
#         self.mlp = th.nn.Sequential(
#         th.nn.Linear(in_dim, hid), th.nn.ReLU(), th.nn.Dropout(dropout),
#         th.nn.Linear(hid, out_dim)
#         )
#         self.propa = APPNPConv(k, alpha, edge_drop=0.0)
#         self.dropout = th.nn.Dropout(dropout)
#
#
#     def forward(self, G: dgl.DGLGraph, x: th.Tensor) -> th.Tensor:
#         h0 = self.mlp(x)
#         h = self.propa(G, h0)
#         return h
class GCNGuard(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, layer_norm_first=False, use_ln=True, attention_drop=True):
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.dropout = dropout
        self.attention_drop = attention_drop
        self.gate = 0.0
        self.prune_edge = True

        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels, add_self_loops=False)])
        self.lns   = nn.ModuleList([nn.LayerNorm(in_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.lns.append(nn.LayerNorm(hidden_channels))
        self.lns.append(nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))

        if self.attention_drop:
            self.drop_learn = nn.Linear(2, 1)

    def reset_parameters(self):
        for c in self.convs: c.reset_parameters()
        for ln in self.lns:  ln.reset_parameters()
        if self.attention_drop: self.drop_learn.reset_parameters()

    @torch.no_grad()
    def att_coef(self, features: Tensor, adj: SparseTensor) -> SparseTensor:
        # 与您上传的实现一致：对每条边计算 cos 相似度，阈值/归一化并补自环，得到新稀疏邻接
        row, col = adj.coo()[:2]
        n_total = features.size(0)
        if features.size(1) > 512 or row.size(0) > 5e5:
            batch_size = int(1e8 // max(1, features.size(1)))
            bepoch = row.size(0) // batch_size + (row.size(0) % batch_size > 0)
            sims = []
            for i in range(bepoch):
                st, ed = i * batch_size, min((i + 1) * batch_size, row.size(0))
                sims.append(F.cosine_similarity(features[row[st:ed]], features[col[st:ed]]))
            sims = torch.cat(sims, dim=0)
        else:
            sims = F.cosine_similarity(features[row], features[col])

        mask = torch.logical_or(sims >= 0.1, row == col)  # 阈值=0.1
        row, col, sims = row[mask], col[mask], sims[mask]

        has_self = (row == col).sum().item()
        if has_self:
            sims[row == col] = 0

        deg = scatter_add(sims, row, dim=0, dim_size=n_total)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        sims = deg_inv_sqrt[row] * sims * deg_inv_sqrt[col]

        deg_new = scatter_add(torch.ones_like(sims), col, dim=0, dim_size=n_total) + 1
        deg_inv_sqrt_new = deg_new.float().pow_(-1.0)
        deg_inv_sqrt_new.masked_fill_(deg_inv_sqrt_new == float('inf'), 0)

        if has_self == 0:
            new_idx = torch.arange(n_total, device=row.device)
            row = torch.cat((row, new_idx), 0)
            col = torch.cat((col, new_idx), 0)
            sims = torch.cat((sims, deg_inv_sqrt_new), 0)
        elif has_self < n_total:
            new_idx = torch.ones(n_total, device=row.device, dtype=torch.bool)
            new_idx[row[row == col]] = False
            new_idx = torch.nonzero(new_idx, as_tuple=True)[0]
            row = torch.cat((row, new_idx), 0)
            col = torch.cat((col, new_idx), 0)
            sims = torch.cat((sims, deg_inv_sqrt_new[new_idx]), 0)
            sims[row == col] = deg_inv_sqrt_new
        else:
            sims[row == col] = deg_inv_sqrt_new

        sims = sims.exp()
        return SparseTensor(row=row, col=col, value=sims, sparse_sizes=(n_total, n_total))

    def forward(self, x: Tensor, adj: SparseTensor):
        if self.layer_norm_first:
            x = self.lns[0](x)
        adj_mem = None
        for i, conv in enumerate(self.convs[:-1]):
            if self.prune_edge:
                new_adj = self.att_coef(x, adj)
                if adj_mem is not None and self.gate > 0:
                    dense = self.gate * adj_mem.to_dense() + (1 - self.gate) * new_adj.to_dense()
                    row, col = dense.nonzero()[:2]
                    adj_values = dense[row, col]
                    edge_index = torch.stack((row, col), dim=0)
                    adj = new_adj  # 继续用最新
                else:
                    adj_mem = new_adj
                    row, col, adj_values = adj_mem.coo()[:3]
                    edge_index = torch.stack((row, col), dim=0)
            else:
                row, col, adj_values = adj.coo()[:3]
                edge_index = torch.stack((row, col), dim=0)

            x = conv(x, edge_index, edge_weight=adj_values)
            if self.use_ln: x = self.lns[i+1](x)
            x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层
        new_adj = self.att_coef(x, adj) if self.prune_edge else adj
        row, col, adj_values = new_adj.coo()[:3]
        edge_index = torch.stack((row, col), dim=0)
        x = self.convs[-1](x, edge_index, edge_weight=adj_values)
        return x.log_softmax(dim=-1)
# 参考实现来源：您上传的 model_pyg.py（GCNGuard 的裁边+归一化流程）。

# ===============================================================
# RGAT（鲁棒 GAT）：使用 RGATConv（在消息时对注意力做阈值化/softmax）
# ===============================================================
class RGATConv(GATConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int,
                 heads: int = 1, threshold: float = 0.1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, att_cpu: bool = False,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, bias, **kwargs)
        self.threshold = threshold
        self.att_cpu = att_cpu

    def message(self, x_j: Tensor, raw_x: Tensor, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        # 自定义注意力：基于原始特征的 cosine，相似度<阈值则近似屏蔽（~0权重），再做 softmax
        # 为了与原实现一致，这里复用 self._alpha 的机制（见您上传的代码）
        if isinstance(raw_x, Tensor):
            x_i = raw_x[index]
            x_j_raw = raw_x[self.edge_index[1]] if hasattr(self, 'edge_index') else x_j  # 兜底
        else:
            x_i, x_j_raw = raw_x

        raw_i = x_i if x_i.dim() == 2 else x_i.squeeze(1)
        raw_j = x_j_raw if x_j_raw.dim() == 2 else x_j_raw.squeeze(1)

        if self.att_cpu:
            device = raw_i.device
            raw_i = raw_i.cpu(); raw_j = raw_j.cpu()

        alpha = F.cosine_similarity(raw_i, raw_j, dim=-1).unsqueeze(1)
        alpha[alpha < self.threshold] = 1e-6  # 近似屏蔽
        alpha = softmax(alpha.log(), index, ptr, size_i)

        if self.att_cpu:
            alpha = alpha.to(x_j.device)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index: Union[Tensor, SparseTensor],
                size: Tuple[Optional[int], Optional[int]] = None, return_attention_weights=None):
        # 直接复用父类的 propagate 流程，但在 message 阶段使用上面的阈值注意力
        if isinstance(edge_index, SparseTensor):
            edge_index = set_diag(edge_index) if self.add_self_loops else edge_index
        else:
            num_nodes = x.size(0) if isinstance(x, Tensor) else x[0].size(0)
            edge_index, _ = remove_self_loops(edge_index)
            if self.add_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=(self.lin_l(x) if isinstance(x, Tensor) else self.lin_l(x[0]),
                                            None if not isinstance(x, tuple) else self.lin_r(x[1])),
                             raw_x=x, size=size)
        alpha = self._alpha; self._alpha = None
        out = out.view(-1, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
        if self.bias is not None: out = out + self.bias
        return out

class RGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5,
                 layer_norm_first=False, use_ln=True, threshold=0.1, heads=1, att_dropout=0.0, att_cpu=False):
        super().__init__()
        if use_ln is False:
            warnings.warn("RGAT 通常建议搭配 LayerNorm。")
        self.layer_norm_first = layer_norm_first
        self.use_ln = True
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.lns   = nn.ModuleList()
        self.convs.append(RGATConv(in_channels, hidden_channels // heads, heads=heads,
                                   threshold=threshold, dropout=att_dropout, att_cpu=att_cpu))
        self.lns.append(nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGATConv(hidden_channels, hidden_channels // heads, heads=heads,
                                       threshold=threshold, dropout=att_dropout, att_cpu=att_cpu))
            self.lns.append(nn.LayerNorm(hidden_channels))
        self.lns.append(nn.LayerNorm(hidden_channels))
        self.convs.append(RGATConv(hidden_channels, out_channels, dropout=att_dropout, att_cpu=att_cpu))

    def reset_parameters(self):
        for c in self.convs: c.reset_parameters()
        for ln in self.lns:  ln.reset_parameters()

    def forward(self, x: Tensor, adj_t: Union[Tensor, SparseTensor]):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln: x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


# 参考实现与默认值来源：您上传的 model_pyg.py/gnn_misg.py（RGAT 和 RGATConv，默认 threshold=0.1；脚本里常把 att_dropout 设为 0.0）。

# ===============================================================
# FLAG（特征扰动版的对抗训练步骤，常用的“Free/Projected”风格）
# ===============================================================

# 使用方法：
#   loss = flag_train_step(model, data.x, data.adj_t, data.y, train_idx, optimizer, m=3, step_size=1e-3)
# 若您想复现实验脚本里的“更强 FLAG”，直接把 m 调大即可（例如 m=30/50/100）。
