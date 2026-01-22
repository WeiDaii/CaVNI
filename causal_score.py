# c2sni/causal_score.py
from typing import Optional
import torch
import dgl

class CausalScorer:
    def __init__(self, victim, device, alpha: float = 0.6, drop_top: int = 5):
        self.victim = victim
        self.device = device
        self.alpha = alpha
        self.drop_top = int(drop_top)  # ← 新增：每个节点“本地反事实”要切掉的关键邻居条数

    @torch.inference_mode()
    def score_vulnerability(self, G: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor,
                            node_idx: torch.Tensor, topk: Optional[int] = None) -> torch.Tensor:
        """
        返回每个节点的“局部反事实效应”得分：切掉与其最关键的若干条边后，
        该节点预测置信度的下降量。topk 控制每个节点最多考虑多少个最关键邻居。
        """
        G = G.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)

        node_idx = node_idx.to(self.device).view(-1).long()
        M = node_idx.shape[0]

        # 先对 node_idx 里的节点逐一打分
        local_scores = torch.zeros(M, device=self.device, dtype=torch.float32)
        for i, u in enumerate(node_idx.tolist()):
            s = self._local_cf_effect(G, x, int(u), drop_top=self.drop_top)
            local_scores[i] = s

        # 把分数放回“全图”向量（未参与评估的节点设为 -inf，保证后续 topk 只会从 node_idx 中选）
        N = G.num_nodes()
        scores_full = torch.full((N,), float('-inf'), device=self.device)
        scores_full[node_idx] = local_scores

        # 选择 Top-K 作为锚点（全局ID）
        K = M if topk is None else min(int(topk), M)
        top_nodes = torch.topk(scores_full, k=K, largest=True).indices  # 全局ID

        return {
            "scores": scores_full,  # [N]，其余为 -inf
            "top_nodes": top_nodes  # [K]，按易损度降序
        }

    # ---------------- 内部：兼容获取 eid 的小工具 ----------------
    def _safe_edge_ids(self, G: dgl.DGLGraph, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        只为存在的 (src, dst) 取 eid。不存在的边自动过滤。
        兼容不同 DGL 版本的行为。
        """
        # 设备
        try:
            dev = G.device
        except Exception:
            # 从 ndata 推断
            dev = next((v.device for v in G.ndata.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))

        src = torch.as_tensor(src, device=dev)
        dst = torch.as_tensor(dst, device=dev)
        if src.dim() == 0 and dst.dim() >= 1:
            src = src.repeat(dst.shape[0])

        # 过滤不存在的边
        exists = G.has_edges_between(src, dst)
        if exists.ndim > 0:
            if exists.sum() == 0:
                return torch.zeros(0, dtype=torch.int64, device=dev)
            src = src[exists]
            dst = dst[exists]
        elif not bool(exists):
            return torch.zeros(0, dtype=torch.int64, device=dev)

        # 取 eid
        eids = G.edge_ids(src, dst)
        if isinstance(eids, tuple):   # 旧版本某些情况会返回 (eids,)
            eids = eids[0]
        return eids

    @torch.inference_mode()
    def _local_cf_effect(self, G: dgl.DGLGraph, x: torch.Tensor, node: int, drop_top: int = 5) -> float:
        """
        反事实效应：切掉该节点最“关键”的若干邻居边后，其预测置信度下降量。
        “关键”邻居这里仍按你原逻辑（比如度/注意力/ppMI 等）提供的 cut。
        这里我们只修正 eid 获取与删边的实现，避免 dgl.edge_ids 报错。
        """
        # 1) 原始置信度
        p0 = self.victim.predict_proba(G, x, nodes=torch.tensor([node], device=x.device)).item()

        # 2) 选择要切的邻居（保持你原来的选择逻辑；这里给个稳妥的默认：按度高的邻居）
        nbrs = G.successors(node)     # 出邻居；如需无向/入邻，可换 predecessors 或 union
        if nbrs.numel() == 0:
            return 0.0
        # 简单按邻居度排序（可替换为你原先的打分）
        deg = G.out_degrees(nbrs)
        topk = min(drop_top, int(nbrs.shape[0]))
        _, idx = torch.topk(deg, k=topk, largest=True)
        cut = nbrs[idx]   # 要切掉的邻居集合 (node -> cut)

        # 3) 取 eid 并删边
        eids = self._safe_edge_ids(G, torch.full_like(cut, node), cut)
        if eids.numel() == 0:
            return 0.0
        H = dgl.remove_edges(G, eids)

        # 4) 删边后的置信度
        p1 = self.victim.predict_proba(H, x, nodes=torch.tensor([node], device=x.device)).item()

        # 5) 反事实效应（下降越大越脆弱）
        return max(0.0, p0 - p1)
