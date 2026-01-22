
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from .utils_guard import gcn_norm


class RobustGCNConv(nn.Module):
    r"""
    Description
    -----------
    RobustGCN convolutional layer.
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    act0 : func of torch.nn.functional, optional
        Activation function. Default: ``F.elu``.
    act1 : func of torch.nn.functional, optional
        Activation function. Default: ``F.relu``.
    initial : bool, optional
        Whether to initialize variance.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.
    """

    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        self.dropout = dropout

    def reset_parameters(self):
        self.mean_conv.reset_parameters()
        self.var_conv.reset_parameters()

    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""
        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        """
        if self.initial:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var = mean
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)
        else:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var = F.dropout(var, p=self.dropout, training=self.training)
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var) + 1e-6  # avoid abnormal gradient
            attention = torch.exp(-var)

            mean = mean * attention
            var = var * attention * attention
            mean = adj0 @ mean
            var = adj1 @ var

        return mean, var

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
import torch_geometric.utils as utils


class RobustGCN(nn.Module):
    r"""
    Description
    -----------
    Robust Graph Convolutional Networks (`RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__)
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    dropout : bool, optional
        Whether to dropout during training. Default: ``True``.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
    # def __init__(self, in_features, out_features, hidden_features, dropout=True):
        super(RobustGCN, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(in_channels, hidden_channels, act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(num_layers - 2):
            self.layers.append(RobustGCNConv(hidden_channels, hidden_channels,
                                             act0=self.act0, act1=self.act1, dropout=dropout))
        self.layers.append(RobustGCNConv(hidden_channels, out_channels, act0=self.act0, act1=self.act1))
        self.dropout = dropout
        self.use_ln = True
        self.gaussian = None
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        r"""
        x : torch.Tensor
        adj : torch_sparse.SparseTensor or compatible
        """
        # 邻接放到同一 device
        from torch_sparse import SparseTensor
        if isinstance(adj, SparseTensor):
            adj = adj.to(x.device)

        # 归一化后的邻接也放到 GPU
        adj0 = gcn_norm(adj).to(x.device)
        adj1 = gcn_norm(adj, order=-1.0).to(x.device)

        mean, var = x, x
        for layer in self.layers:
            # 用“位置参数”调用，兼容不同 forward 签名
            mean, var = layer(mean, var, adj0, adj1)

        # 采样保持在同一 device
        sample = torch.randn_like(var, device=x.device)
        out = mean + sample * var.clamp_min(1e-12).sqrt()

        return F.log_softmax(out, dim=-1)
