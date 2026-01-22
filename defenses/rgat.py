
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from .utils_guard import gcn_norm
import warnings

from math import degrees
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_scatter.scatter import scatter_add
import warnings
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,GATv2Conv
from torch_sparse.tensor import SparseTensor
from typing import Union, Tuple, Optional, List, Dict
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from torch_sparse import SparseTensor  # 如需
from torch import Tensor
class RGATConv(GATConv):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, threshold=0.1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0., att_cpu=False,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(RGATConv, self).__init__(in_channels, out_channels, heads,
                                       concat, negative_slope, dropout, add_self_loops, bias, **kwargs)
        self.threshold = threshold
        self.att_cpu = att_cpu
        # print(self.__dict__)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        # alpha_l: OptTensor = None
        # alpha_r: OptTensor = None

        raw_x = x
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `RGATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            # alpha_l = (x_l * self.att_l).sum(dim=-1)
            # alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `RGATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            # alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                # alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        # assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             raw_x=raw_x, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor,
                raw_x_i: OptTensor, raw_x_j: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # print(raw_x_i.size(),raw_x_j.size())
        # with torch.no_grad():
        #     alpha_sim = F.cosine_similarity(raw_x_i, raw_x_j,dim=-1).unsqueeze(1)
        #     # alpha_sim[alpha_sim>0.1] = 1.0
        #     alpha_sim[alpha_sim<0.1] = 0
        if self.att_cpu:
            print("move vars to cpu")
            device = raw_x_i.device
            raw_x_i = raw_x_i.cpu()
            raw_x_j = raw_x_j.cpu()
        if raw_x_i.size(1) <= 500:
            alpha = F.cosine_similarity(raw_x_i, raw_x_j, dim=-1).unsqueeze(1)
            alpha[alpha < self.threshold] = 1e-6
        else:
            alpha = F.cosine_similarity(x_i.squeeze(1), x_j.squeeze(1), dim=-1).unsqueeze(1)
            alpha[alpha < 0.5] = 1e-6
        # att = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = softmax(alpha.log(), index, ptr, size_i)
        # alpha[alpha<0.5] -= 0.2
        # alpha = F.leaky_relu(alpha,self.negative_slope)
        # alpha = alpha_sim * alpha
        # alpha = softmax(alpha, index, ptr, size_i)
        if self.att_cpu:
            # device = raw_x_i.device
            raw_x_i = raw_x_i.to(device)
            raw_x_j = raw_x_j.to(device)
            alpha = alpha.to(device)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class RGAT(nn.Module):
    """
    Robust GAT inspired by GNNGuard
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, threshold=0.1, heads=1, att_dropout=0.6, att_cpu=False):
        super(RGAT, self).__init__()
        self.layer_norm_first = layer_norm_first
        if use_ln==False:
            warnings.warn("RGAT has to be accompanied with LN inside")
        self.use_ln = True
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGATConv(in_channels, hidden_channels//heads, heads=heads, threshold=threshold, dropout=att_dropout, att_cpu=att_cpu))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGATConv(hidden_channels, hidden_channels//heads, heads=heads, threshold=threshold, dropout=att_dropout, att_cpu=att_cpu))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(RGATConv(hidden_channels, out_channels, dropout=att_dropout, att_cpu=att_cpu))

        self.dropout = dropout
        self.threshold = threshold

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, PairTensor,Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_sparse import SparseTensor, set_diag
