import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GraphConv, GINConv, APPNPConv, SAGEConv, SGConv



# dgl_hang_constant.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv


# dgl_evennet_consistent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# dgl_jaccard_and_simpgcn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# -------------------------- utils --------------------------

def _norm_adj_with_selfloop(g, device):
    """ \hat{A} = D^{-1/2} (A + I) D^{-1/2}  (sparse COO) """
    gg = dgl.add_self_loop(dgl.remove_self_loop(g))
    N = gg.num_nodes()
    src, dst = gg.edges()
    deg = gg.in_degrees().float().clamp(min=1).to(device)
    d_isqrt = deg.pow(-0.5)
    vals = d_isqrt[src] * d_isqrt[dst]
    idx = torch.stack([src.to(device), dst.to(device)], dim=0)
    Ahat = torch.sparse_coo_tensor(idx, vals, (N, N), device=device).coalesce()
    return Ahat

def _row_norm_coo(spmat):
    """ row-normalize a sparse COO matrix (sum=1 per row) """
    idx = spmat.indices(); vals = spmat.values(); N = spmat.size(0)
    row = idx[0]
    row_sum = torch.zeros(N, device=vals.device).index_add_(0, row, vals)
    row_sum = row_sum.clamp(min=1e-12)
    new_vals = vals / row_sum[row]
    return torch.sparse_coo_tensor(idx, new_vals, spmat.size(), device=vals.device).coalesce()

# -------------------------- Jaccard-GCN --------------------------

class dgl_jaccard_gcn(nn.Module):
    """
    GCN-Jaccard (IJCAI'19 防御版 GCN)
    运行逻辑：在前向的第一次调用时，按阈值 τ 过滤“低相似边”，再在裁剪后的图上做两层 GCN。
    - 二值特征：Jaccard；连续特征：Cosine（论文常见做法）；
      默认 τ=0（和原文“只删 Jaccard=0 的边”一致）。 参见 IJCAI'19、防御设置与 GANI 实验说明。
    """
    def __init__(self, input_dim, hidden, classes,
                 thresh: float = 0.0, metric: str = "auto",
                 input_dropout: float = 0.0, dropout: float = 0.5):
        super().__init__()
        from dgl.nn.pytorch import GraphConv
        self.gc1 = GraphConv(input_dim, hidden, norm='both', allow_zero_in_degree=True)
        self.gc2 = GraphConv(hidden, classes, norm='both', allow_zero_in_degree=True)
        self.idrop = nn.Dropout(input_dropout) if input_dropout>0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.thresh = float(thresh)
        self.metric = metric  # 'auto' | 'jaccard' | 'cosine'
        self._cached = None   # (n,e,device) -> pruned graph

    @torch.no_grad()
    def _prune_graph(self, g, x):
        device = x.device
        key = (g.num_nodes(), g.num_edges(), device)
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]

        # choose metric
        use_cos = False
        if self.metric == "cosine":
            use_cos = True
        elif self.metric == "jaccard":
            use_cos = False
        else:  # auto
            is_binary = torch.all((x==0) | (x==1))
            use_cos = not is_binary

        gg = dgl.remove_self_loop(g).to(device)
        src, dst = gg.edges()
        Xs, Xt = x[src], x[dst]

        if use_cos:
            # cosine similarity
            num = (Xs * Xt).sum(dim=1)
            den = (Xs.norm(dim=1) * Xt.norm(dim=1)).clamp(min=1e-12)
            sim = num / den
        else:
            # jaccard over binary features
            Xs_b = Xs.bool(); Xt_b = Xt.bool()
            inter = (Xs_b & Xt_b).sum(dim=1).float()
            union = (Xs_b | Xt_b).sum(dim=1).float().clamp(min=1e-12)
            sim = inter / union

        mask = sim >= self.thresh  # 原文默认 τ=0，仅删除相似度为0的边
        pruned = dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes(), device=device)
        pruned = dgl.to_simple(pruned) # 去重
        pruned = dgl.add_self_loop(dgl.remove_self_loop(pruned))
        self._cached = (key, pruned)
        return pruned

    def forward(self, g, features):
        g2 = self._prune_graph(g, features)
        x = self.idrop(features)
        x = F.relu(self.gc1(g2, x))
        x = self.drop(x)
        x = self.gc2(g2, x)
        return x

# -------------------------- SimP-GCN --------------------------

class dgl_simpgcn(nn.Module):
    """
    SimP-GCN (WSDM'21) —— 节点相似性保持的 GCN
    逐层传播矩阵（Eq.(9)(11)）:
        p^(l) = diag(s^(l)) * Â + diag(1 - s^(l)) * S_f
        \tilde{p}^(l) = p^(l) + γ * diag(K^(l))
      其中 s^(l) = sigmoid(H^(l-1) w_s + b_s)，K^(l) = H^(l-1) W_K + b_K（Eq.(12)）
      最终 H^(l) = σ( \tilde{p}^(l) H^(l-1) W^(l) )（Eq.(13)）
    自监督损失（Eq.(15)(16)）:
        L = L_class + λ * L_self，L_self 对“最相似/最不相似”成对样本做回归（kNN 由 Eq.(8) 余弦构造，论文默认 k=20）。
    """
    def __init__(self,
                 input_dim: int, hidden: int, classes: int,
                 num_layers: int = 2,
                 k: int = 20,                 # Eq.(8) 默认
                 gamma: float = 0.1,          # Eq.(11) γ
                 lambda_self: float = 1e-3,   # Eq.(16) λ
                 m_pairs: int = 5,            # 每个节点选 m 个最相似 & m 个最不相似
                 input_dropout: float = 0.0,
                 dropout: float = 0.5,
                 bn: bool = False):
        super().__init__()
        assert num_layers >= 2
        self.k = int(k); self.gamma = float(gamma)
        self.lambda_self = float(lambda_self)
        self.m_pairs = int(m_pairs)

        # 线性映射
        dims = [input_dim] + [hidden]*(num_layers-1) + [classes]
        self.W = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=True) for i in range(num_layers)])
        # s^(l)、K^(l) 的参数（到最后一层之前）
        self.s_mlps = nn.ModuleList([nn.Linear(dims[i], 1) for i in range(num_layers)])
        self.k_mlps = nn.ModuleList([nn.Linear(dims[i], 1) for i in range(num_layers)])

        self.idrop = nn.Dropout(input_dropout) if input_dropout>0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.bn = nn.ModuleList([nn.BatchNorm1d(dims[i+1]) for i in range(num_layers-1)]) if bn else None

        # 自监督：线性映射 f_w( h_i - h_j ) -> 1（Eq.(15) 的线性 map）
        self.pair_lin = nn.Linear(hidden, 1, bias=True)

        # 缓存
        self._Ahat = None      # 结构图 \hat{A}
        self._Sf = None        # 特征 kNN 图 S_f
        self._pairs = None     # 训练用的成对索引 (2, |T|) 以及 S_ij

        # 记录本次前向的 self-loss，供外部加到 CE 上
        self.last_self_loss = torch.tensor(0.0)

    # --------- helpers (构造 \hat{A} 与 kNN 特征图) ---------
    def _get_Ahat(self, g, device):
        if self._Ahat is None or self._Ahat.device != device:
            self._Ahat = _norm_adj_with_selfloop(g, device)
        return self._Ahat

    @torch.no_grad()
    def _build_knn_graph(self, X, k):
        # 余弦相似（Eq.(8)）：先单位化，再点积得到 NxN 相似度，取每行 top-k
        Xn = F.normalize(X, p=2, dim=1)
        S = Xn @ Xn.t()
        N = S.size(0)
        S.fill_diagonal_(-1.0)  # 排除自己
        vals, idx = torch.topk(S, k=k, dim=1, largest=True)
        # 构造 COO：行 i 连接到 idx[i, j]
        rows = torch.arange(N, device=X.device).unsqueeze(1).expand_as(idx).reshape(-1)
        cols = idx.reshape(-1)
        sim = vals.reshape(-1).clamp(min=0)  # 负值置零（只保留相似的）
        A = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), sim, (N, N), device=X.device)
        # 对称化 + 度归一化
        A = (A + A.t()).coalesce()
        D = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12).pow(-0.5)
        r, c = A.indices()
        nvals = D[r] * A.values() * D[c]
        Sf = torch.sparse_coo_tensor(A.indices(), nvals, A.size(), device=X.device).coalesce()
        self._Sf = Sf

    @torch.no_grad()
    def _build_pairs(self, X):
        # 为自监督采样 |T|：每个节点拿 m 个最相似 + m 个最不相似（按余弦, Eq.(8)）
        Xn = F.normalize(X, p=2, dim=1)
        S = Xn @ Xn.t()
        N = S.size(0)
        S.fill_diagonal_(0.0)
        topv, topi = torch.topk(S, k=min(self.m_pairs, N-1), dim=1, largest=True)
        botv, boti = torch.topk(S, k=min(self.m_pairs, N-1), dim=1, largest=False)
        i_top = torch.arange(N, device=X.device).unsqueeze(1).expand_as(topi).reshape(-1)
        j_top = topi.reshape(-1)
        i_bot = torch.arange(N, device=X.device).unsqueeze(1).expand_as(boti).reshape(-1)
        j_bot = boti.reshape(-1)
        pairs_i = torch.cat([i_top, i_bot], dim=0)
        pairs_j = torch.cat([j_top, j_bot], dim=0)
        Sij = torch.cat([topv.reshape(-1), botv.reshape(-1)], dim=0)
        self._pairs = (torch.stack([pairs_i, pairs_j], dim=0), Sij)

    # --------- forward ---------
    def forward(self, g, features):
        x = self.idrop(features)
        device = x.device

        # 构造/缓存 \hat{A} 与 S_f（一次即可）
        Ahat = self._get_Ahat(g, device)
        if self._Sf is None or self._Sf.device != device:
            self._build_knn_graph(x, self.k)
        if self._pairs is None:
            self._build_pairs(x)

        H = x
        H1_store = None  # 用于 L_self
        for l, Wl in enumerate(self.W):
            # s^(l) 和 K^(l)
            s = torch.sigmoid(self.s_mlps[l](H)).squeeze(-1)            # (N,)
            K = self.k_mlps[l](H).squeeze(-1)                           # (N,)

            # 计算  \tilde{p}^(l) H = s * (Â H) + (1-s) * (S_f H) + γ * (K ⊙ H)
            AH = torch.sparse.mm(Ahat, H)
            SfH = torch.sparse.mm(self._Sf, H)
            PH = s.unsqueeze(1) * AH + (1.0 - s).unsqueeze(1) * SfH + self.gamma * (K.unsqueeze(1) * H)

            H = Wl(PH)
            if l < len(self.W) - 1:
                H = F.relu(H)
                if self.bn is not None:
                    H = self.bn[l](H)
                H = self.drop(H)
                if H1_store is None:  # 第一层输出，用于 Eq.(15)
                    H1_store = H

        # 自监督损失（Eq.(15)）
        if H1_store is None:  # num_layers==1 时保护（一般不会）
            self.last_self_loss = torch.tensor(0.0, device=device)
        else:
            (pi, pj), Sij = self._pairs
            diff = H1_store[pi] - H1_store[pj]
            pred = self.pair_lin(diff).squeeze(-1)
            self.last_self_loss = F.mse_loss(pred, Sij, reduction='mean')

        return H  # logits

# -------------------------- 用法小贴士 --------------------------
# 训练 SimP-GCN 时，按论文总损失：
#  loss = F.cross_entropy(logits[train_idx], y[train_idx]) + model.lambda_self * model.last_self_loss
# Jaccard-GCN 则与普通 GCN 一样只用 CE。





def _build_norm_adj_P(g, device):
    """
    P = D^{-1/2} A D^{-1/2} （严格不加自环；与论文/仓库一致）
    参考：EvenNet 论文 §3.2 与官方实现思路。
    """
    g = dgl.remove_self_loop(g)
    N = g.num_nodes()
    src, dst = g.edges()
    src = src.to(device); dst = dst.to(device)
    deg = g.in_degrees().float().clamp(min=1).to(device)
    d_isqrt = deg.pow(-0.5)
    vals = d_isqrt[src] * d_isqrt[dst]
    idx = torch.stack([src, dst], dim=0)                 # COO indices
    P = torch.sparse_coo_tensor(idx, vals, (N, N), device=device).coalesce()
    return P

class dgl_evennet(nn.Module):
    """
    EvenNet（与原仓库一致的 DGL 复刻）
    公式：Z = sum_{k=0}^{K_even} theta[k] * P^{2k} * ell(X)  ->  Linear  -> logits
    只保留偶次项（忽略所有奇数跳邻居），P = D^{-1/2} A D^{-1/2}（不加自环）。
    """
    def __init__(self,
                 input_dim: int,
                 hidden: int,
                 classes: int,
                 K: int = 10,                 # 与仓库脚本常用默认保持一致
                 input_dropout: float = 0.4,  # 常见默认
                 dropout: float = 0.0,
                 bn: bool = False,
                 mlp_layers: int = 1):
        super().__init__()
        assert K >= 0
        self.K = int(K)
        self.K_even = self.K // 2

        # ell(X)：仓库里是线性或轻量 MLP；提供 1/2 层可选
        if mlp_layers == 1:
            self.enc = nn.Linear(input_dim, hidden)
        elif mlp_layers == 2:
            self.enc = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
            )
        else:
            raise ValueError("mlp_layers must be 1 or 2")

        # 偶次滤波系数 θ_k（可学习），k=0..K_even
        self.theta = nn.Parameter(torch.zeros(self.K_even + 1))
        with torch.no_grad():
            self.theta[0] = 1.0   # 从恒等滤波开始，稳健初始化（与官方实现风格一致）

        # 分类头、正则
        self.dec = nn.Linear(hidden, classes)
        self.in_drop  = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.mid_drop = nn.Dropout(dropout)       if dropout       > 0 else nn.Identity()
        self.bn = nn.BatchNorm1d(hidden) if bn else None

        # 稀疏 P 的简易缓存（静态图时避免重复构造）
        self._P = None
        self._cached_n = None
        self._cached_e = None

    def _get_P(self, g, device):
        n, m = g.num_nodes(), g.num_edges()
        if (self._P is not None) and (self._cached_n == n) and (self._cached_e == m) and (self._P.device == device):
            return self._P
        self._P = _build_norm_adj_P(g, device)
        self._cached_n, self._cached_e = n, m
        return self._P

    def forward(self, g, features):
        x = self.in_drop(features)
        h0 = self.enc(x)                        # ell(X)
        if self.bn is not None:
            h0 = self.bn(h0)

        P = self._get_P(g, h0.device)          # 稀疏 COO
        Z = self.theta[0] * h0                  # k=0: 恒等项
        cur = h0
        # 逐次累乘 P^2：cur <- P^2 cur
        for k in range(1, self.K_even + 1):
            cur = torch.sparse.mm(P, torch.sparse.mm(P, cur))
            Z = Z + self.theta[k] * cur

        Z = self.mid_drop(Z)
        logits = self.dec(Z)
        return logits



class _EnergyNet(nn.Module):
    """
    H_net(q, p): 贴合 vanilla HANG 的做法——
    分别用图卷积抽取 q/p 的结构特征，拼接后经 MLP，能量为 0.5 * ||MLP(...)||_2^2（对所有节点与通道求和）
    最终标量 H 供 autograd 求偏导。
    """
    def __init__(self, hidden: int, energy_hidden: int = None, activation: str = "tanh"):
        super().__init__()
        self.q_conv = GraphConv(hidden, hidden, norm='both', allow_zero_in_degree=True)
        self.p_conv = GraphConv(hidden, hidden, norm='both', allow_zero_in_degree=True)
        eh = energy_hidden or hidden
        self.fc1 = nn.Linear(2 * hidden, eh)
        self.fc2 = nn.Linear(eh, hidden)   # 输出维度与 hidden 对齐，便于做 L2 范数
        self.act = torch.tanh if activation == "tanh" else F.elu

    def forward(self, g, q, p):
        qg = self.q_conv(g, q)     # [N, H]
        pg = self.p_conv(g, p)     # [N, H]
        z  = torch.cat([qg, pg], dim=-1)   # [N, 2H]
        h  = self.act(self.fc1(z))         # [N, eh]
        h  = self.fc2(h)                   # [N, H]
        H  = 0.5 * (h.pow(2).sum())        # 标量能量
        return H


class _HANGBlock(nn.Module):
    """
    单步 HANG（Euler），常量块（权重可复用）：
        q_{t+1} = q_t + h * ∂H/∂p  (+ source_q)
        p_{t+1} = p_t - h * ∂H/∂q  (+ source_p)
    """
    def __init__(self, hidden: int, step_size: float = 1.0,
                 add_source: bool = False, batch_norm: bool = False,
                 energy_hidden: int = None, activation: str = "tanh"):
        super().__init__()
        self.h = float(step_size)
        self.energy_net = _EnergyNet(hidden, energy_hidden, activation)
        self.src_q = nn.Linear(hidden, hidden) if add_source else None
        self.src_p = nn.Linear(hidden, hidden) if add_source else None
        self.bn = nn.BatchNorm1d(hidden) if batch_norm else None

    def forward(self, g, q, p, create_graph: bool):


        # # 需可导以便 autograd 计算偏导
        # q = q.requires_grad_(True)
        # p = p.requires_grad_(True)
        #
        # H = self.energy_net(g, q, p)                # 标量
        # grad_q, grad_p = torch.autograd.grad(       # ∂H/∂(q,p)
        #     H, (q, p), create_graph=create_graph
        # )
        #
        # dq = grad_p
        # dp = -grad_q
        #
        # if self.src_q is not None:
        #     dq = dq + self.src_q(q)
        # if self.src_p is not None:
        #     dp = dp + self.src_p(p)
        if create_graph is None:
            create_graph = self.training

            # ⬇️ 即使外层在 no_grad，这里也强制打开 Autograd
        with torch.enable_grad():
            # ⬇️ 确保 q/p 是 leaf 且需要梯度（避免 no_grad 导致无 grad_fn）
            q = q.detach().clone().requires_grad_(True)
            p = p.detach().clone().requires_grad_(True)

            # 你的哈密顿量/能量函数（返回标量或 [N] 向量）
            H = self.energy_net(g, q, p)  # 或 self.energy(g, q, p)
            if H.dim() != 0:  # autograd.grad 需要标量
                H = H.sum()
            if not H.requires_grad:
                raise RuntimeError(
                    "H does not require grad — likely still under no_grad/inference_mode "
                    "somewhere up the call stack. Remove decorators and use conditional context."
                )
            grad_q, grad_p = torch.autograd.grad(
                H, (q, p),
                create_graph=create_graph,  # 训练时 True，评估时 False
                retain_graph=True
            )
            dq = grad_p
            dp = -grad_q
            # 接下来按你原来的方式用梯度更新 q/p
        # q_next = q + self.step_size * grad_p
        # p_next = p - self.step_size * grad_q
        # Euler 更新
        q_next = q + self.h * dq
        p_next = p + self.h * dp

        if self.bn is not None:
            q_next = self.bn(q_next)
        return q_next, p_next


class dgl_hang(nn.Module):
    """
    与 dgl_sgc 相同接口：
        logits = model(g, features)

    关键超参（与源码脚本/README 对齐）：
      - hidden: 隐层维度（= --hidden_dim）
      - time, step_size: 连续时间与步长，steps = round(time/step_size)（= --time, --step_size）
      - add_source: 是否加入源项（= --add_source）
      - input_dropout / dropout / batch_norm（= --input_dropout / --dropout / --batch_norm）
      - energy_hidden: H_net 的中间维度（默认与 hidden 相同）
      - activation: H_net 激活（默认 tanh，更贴近 vanilla HANG）
      - p_init: 'copy' | 'zero' —— p0 初始化策略（源码常用 copy）
      - share_block: 是否“常量块”参数共享（源码常用 constant）
    """
    def __init__(self,
                 input_dim: int,
                 hidden: int,
                 classes: int,
                 time: float = 3.0,
                 step_size: float = 1.0,
                 add_source: bool = False,
                 input_dropout: float = 0.0,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 energy_hidden: int = None,
                 activation: str = "tanh",
                 p_init: str = "copy",
                 share_block: bool = True):
        super().__init__()
        assert p_init in ("copy", "zero")
        self.p_init = p_init
        self.steps = max(1, int(round(time / step_size)))
        self.create_graph_training = True  # 训练时构建高阶图；推理时会自动关掉以省显存
        self.requires_grad_forward = True
        # 编码 & 分类头
        self.enc = nn.Linear(input_dim, hidden)
        self.dec = nn.Linear(hidden, classes)

        # 正则
        self.in_drop  = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.mid_drop = nn.Dropout(dropout)       if dropout       > 0 else nn.Identity()

        # 常量块 or 非共享堆叠
        if share_block:
            self.block = _HANGBlock(hidden, step_size, add_source, batch_norm, energy_hidden, activation)
        else:
            self.blocks = nn.ModuleList([
                _HANGBlock(hidden, step_size, add_source, batch_norm, energy_hidden, activation)
                for _ in range(self.steps)
            ])

    @staticmethod
    def _with_self_loops(g):
        # 贴近 PyG.GCNConv：确保 A+I
        if g.num_edges() == 0:
            return dgl.add_self_loop(g)
        g2 = dgl.remove_self_loop(g)
        g2 = dgl.add_self_loop(g2)
        return g2

    def forward(self, g, features):
        # 输入与编码
        g = self._with_self_loops(g)                 # 与源码中 GCNConv 默认自环对齐
        x = self.in_drop(features)
        q = self.enc(x)
        p = q.clone() if self.p_init == "copy" else torch.zeros_like(q)

        # 训练/推理的 autograd 策略
        create_graph = self.training and self.create_graph_training

        # HANG 演化（Euler）
        if hasattr(self, "block"):  # constant-block
            for _ in range(self.steps):
                q, p = self.block(g, q, p, create_graph=create_graph)
                q = self.mid_drop(q)
        else:                       # non-shared blocks
            for blk in self.blocks:
                q, p = blk(g, q, p, create_graph=create_graph)
                q = self.mid_drop(q)

        # 分类头：用 q(T)
        logits = self.dec(q)
        return logits


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


class dgl_gcn(nn.Module):
    def __init__(self, input_dim, nhidden, nclasses):
        super(dgl_gcn, self).__init__()
        self.layer1 = GraphConv(in_feats=input_dim, out_feats=nhidden, allow_zero_in_degree=True)
        self.layer2 = GraphConv(in_feats=nhidden, out_feats=nclasses, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)

        return x


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

