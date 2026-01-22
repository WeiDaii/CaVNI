# -*- coding: utf-8 -*-
"""
受害模型封装：GCN/GAT
"""
import torch, torch.nn as nn, torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from utils import accuracy
from DGL_models import dgl_gat, dgl_gcn, dgl_sage, dgl_appnp, dgl_gin, dgl_sgc, dgl_hang,dgl_evennet,dgl_simpgcn,dgl_jaccard_gcn
from GEN_utils  import load_data_citation, random_splits
from dgl.nn.pytorch.conv import GATConv, GraphConv, GINConv, APPNPConv, SAGEConv, SGConv

class dgl_gcn(nn.Module):
    def __init__(self, input_dim, nhidden, nclasses):
        super(dgl_gcn, self).__init__()
        self.layer1 = GraphConv(in_feats=input_dim, out_feats=nhidden, allow_zero_in_degree=True)
        self.layer2 = GraphConv(in_feats=nhidden, out_feats=nclasses, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)

        return x

class dgl_gat(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads, num_classes, dropout):
        super(dgl_gat, self).__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.layer1 = GATConv(input_dim, out_dim, num_heads[0], feat_drop=dropout[0], attn_drop=dropout[1],
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)
        self.layer2 = GATConv(num_heads[0] * out_dim, num_classes, num_heads[1], feat_drop=dropout[0], attn_drop=dropout[1],
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)

    def forward(self, graph, feat):
        x1 = self.layer1(graph, feat)  # input_dim * num_heads[0] * out_dim
        x1 = x1.flatten(1)
        x1 = F.elu(x1)

        x1 = self.layer2(graph, x1)
        x1 = x1.squeeze(1)
        if self.num_heads[1] > 1:
            x1 = torch.mean(x1, dim=1)
        x1 = F.elu(x1)

        return x1
class dgl_sage(nn.Module):
    def __init__(self, input_dim, nhidden, aggregator_type, nclasses):
        super(dgl_sage, self).__init__()
        self.layer1 = SAGEConv(in_feats=input_dim, out_feats=nhidden, aggregator_type=aggregator_type)
        self.layer2 = SAGEConv(in_feats=nhidden, out_feats=nclasses, aggregator_type=aggregator_type)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        return x


class dgl_appnp(nn.Module):
    def __init__(self, input_dim, hidden, classes, k, alpha):
        super(dgl_appnp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.layer1 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.layer2 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

    def forward(self, g, features):
        features = self.fc1(features)
        x = self.layer1(g, features)
        x = F.elu(self.fc2(x))
        x = self.layer2(g, x)
        x = F.elu(x)
        return x


class dgl_gin(nn.Module):
    def __init__(self, input_dim, hidden, classes, aggregator_type):
        super(dgl_gin, self).__init__()
        self.apply_func1 = nn.Linear(input_dim, hidden)
        self.apply_func2 = nn.Linear(hidden, classes)
        self.layer1 = GINConv(apply_func=self.apply_func1, aggregator_type=aggregator_type)
        self.layer2 = GINConv(apply_func=self.apply_func2, aggregator_type=aggregator_type)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.apply_func1.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_func2.weight, gain=gain)

    def forward(self, g_list, features):
            g = g_list
            x = self.layer1(g, features)
            x = F.elu(x)
            x = self.layer2(g, x)
            x = F.elu(x)
            return x


class dgl_sgc(nn.Module):
    def __init__(self, input_dim, hidden, classes):
        super(dgl_sgc, self).__init__()
        self.layer1 = SGConv(in_feats=input_dim, out_feats=hidden, cached=False, allow_zero_in_degree=True)  # k=1
        self.layer2 = SGConv(in_feats=hidden, out_feats=classes, cached=False, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = F.elu(x)
        x = self.layer2(g, x)
        return x

class Victim:
    def __init__(self, model_name, in_dim, n_classes, lr, wd, args, device):
        self.device = device
        a = args
        if model_name=='gat':
            self.model = dgl_gat(
                input_dim=in_dim, out_dim=a.hidden,
                num_heads=[a.num_heads1, a.num_heads2], num_classes=n_classes,
                dropout=[a.feat_dropout, a.attn_dropout]
            ).to(device)
        if model_name=='gcn':
            self.model = dgl_gcn(in_dim, nhidden=a.hidden, nclasses=n_classes).to(device)
        if model_name.lower() == 'sage':
            self.model = dgl_sage(input_dim=in_dim, nhidden=a.hidden, aggregator_type=a.sage_agg_type, nclasses=n_classes).to(device)
        if model_name.lower() == 'appnp':
            self.model = dgl_appnp(input_dim=in_dim, hidden=a.appnp_hidden, classes=n_classes, k=a.K, alpha=a.alpha).to(device)
        if model_name.lower() == 'gin':
            self.model = dgl_gin(input_dim=in_dim, hidden=a.gin_hidden, classes=n_classes, aggregator_type=a.gin_agg_type).to(device)
        if model_name.lower() == 'sgc':
            self.model = dgl_sgc(input_dim=in_dim, hidden=a.hidden, classes=n_classes).to(device)
        if model_name.lower() == 'simpgcn':
            self.model = dgl_simpgcn(input_dim=in_dim, hidden=a.hidden, classes=n_classes).to(device)
        if model_name.lower() == 'jaccard_gcn':
            self.model = dgl_jaccard_gcn(input_dim=in_dim, hidden=a.hidden, classes=n_classes).to(device)
        if model_name.lower() == 'hang':
            self.model = dgl_hang(input_dim=in_dim,
                hidden=a.hidden,              # = --hidden_dim
                classes=n_classes,
                time=3.0,                  # = --time
                step_size=1.0,        # = --step_size
                add_source=True,      # = --add_source
                input_dropout=0.4,# = --input_dropout
                dropout=0.0,            # = --dropout
                batch_norm=True,      # = --batch_norm
                energy_hidden=None,
                activation='tanh',            # 贴近 vanilla HANG
                p_init='copy',                # p0 = q0（源码常用）
                share_block=True              # 等价于 --block constant（参数跨步共享）
    ).to(device)
        if model_name.lower() == 'evennet':
            self.model = dgl_evennet(
                input_dim=in_dim,
                hidden=a.hidden,  # --hidden
                classes=n_classes,
                K=10,  # --K  (默认 10)
                input_dropout=0.4,
                dropout=0.0,
                bn=False,
                mlp_layers=1
            ).to(device)
        self.lr, self.wd = lr, wd
        if self.model is None:
            raise RuntimeError("Victim.model was not constructed — check model_name and branches.")
    @torch.no_grad()
    def predict(self, g, x):
        self.model.eval()
        return self.model(g, x)

    def fit_eval(self, g, x, y, tr, va, te, epochs=200, patience=50, verbose=False):
        needs_grad_forward = getattr(self.model, "requires_grad_forward", False)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, wait = 0.0, None, 0
        for ep in range(1, epochs+1):
            self.model.train()
            logits = self.model(g, x)
            loss = F.cross_entropy(logits[tr], y[tr])
            opt.zero_grad(); loss.backward(); opt.step()
            if needs_grad_forward:
                # HANG/GraphCON 等需要前向梯度
                with torch.enable_grad():
                    logits = self.model(g, x)
            else:
                with torch.no_grad():
                    logits = self.model(g, x)
            va_acc = accuracy(logits[va], y[va]);
            te_acc = accuracy(logits[te], y[te])
            # with torch.no_grad():
            #     self.model.eval()
            #     logits = self.model(g, x)
            #     va_acc = accuracy(logits[va], y[va]); te_acc = accuracy(logits[te], y[te])
            if verbose and ep%20==0: print(f"  [Victim] ep {ep:3d}/{epochs} | val {va_acc:.4f} | best test {best*100:.2f}%")
            if va_acc > best:
                best, best_state, wait = va_acc, {k:v.clone() for k,v in self.model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience: break
        if best_state: self.model.load_state_dict(best_state)
        with torch.no_grad():
            self.model.eval()
            logits = self.model(g, x)
            te_acc = accuracy(logits[te], y[te])
        return {'test_acc': float(te_acc)}

    @torch.no_grad()
    def logits(self, G, feats):
        """
        统一拿分类 logits（不做 softmax）。
        绝大多数 DGL 模型 forward(G, feats)；如果你的模型是其它签名，改这里即可。
        """
        self.model.eval()
        return self.model(G, feats)

    @torch.no_grad()
    def predict(self, G, feats, nodes=None):
        """
        返回预测标签（argmax）。
        nodes: Tensor[list[int]]，可选；若给定，只取这些节点的预测。
        """
        lg = self.logits(G, feats)
        if nodes is not None:
            lg = lg[nodes]
        return lg.argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(self, G, feats, nodes=None, cls: int = None, reduction: str = "mean"):
        """
        返回置信度（概率）：
          - 如果 cls=None：返回各节点的 *最大类* 概率（模型信心）
          - 如果 cls=k  ：返回第 k 类的概率（可用于 targeted）
        nodes: 仅对这些节点取值（与 scorer 调用保持一致）
        reduction: "none" | "mean"。scorer 里用单节点，返回标量即可。
        """
        lg = self.logits(G, feats)
        if nodes is not None:
            lg = lg[nodes]
        prob = F.softmax(lg, dim=-1)
        if cls is None:
            conf, _ = prob.max(dim=-1)  # 每个节点的最大类概率
        else:
            conf = prob[:, cls]

        if reduction == "none":
            return conf
        if conf.numel() == 1:
            return conf.squeeze()
        return conf.mean()
