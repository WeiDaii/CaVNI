# -*- coding: utf-8 -*-
"""
C2SNI 主入口脚本
论文依据：
- 黑盒 NES: Ilyas et al., ICML'18
- 因果: Pearl '09
"""

# import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import argparse, os, math, random

import numpy as np
import torch
from data_utils import load_dgl_dataset, clone_graph_with_data, set_seed
from victim_models import Victim
from causal_score import CausalScorer
from pruning import prune_edges_by_dissim
from utils import device_of, to_cpu
import dgl
# from attacker_sparse import build_surrogate_from_victim, stage4_sparse_optimize_per_subgraph
# ---------- Robust neighbor choosing helpers ----------
from EdgeSampler import augmstru_enhanced_bernoulli, stage2b_inject_per_subgraph,stage3_connect_between_injected   # 通用增强型 Bernoulli（你已放在 EdgeSampler.py）
# from attacker_sparse import SparseInjectionAttack     # 稀疏矩阵可微优化攻击器（上一条消息给你的实现）
from subgraph_khop import print_khop_subgraph_stats, get_khop_candidates
from utils import extract_anchor_subgraphs_from_G, load_original_graph, merge_anchor_maps, batch_subgraphs_from_injid_to_cands
import torch
# from subgraph_sparse_attack_final import run_subgraph_sparse_attack
# from innov_attack import run_innov_attack
# from subgraph_sparse_attack_modify import run_subgraph_sparse_attack_nes_modify
# from subgraph_sparse_attack import run_subgraph_sparse_attack_nes
# === 兼容预测输出为 [N,C] or [N] 的小工具 ===
from defense_eval import evaluate_defenses

def extract_clean_part_from_adv(G_adv: dgl.DGLGraph, X_adv: torch.Tensor, N0: int):
    """
    从对抗图中抽取前 N0 个原始节点诱导子图（OO 部分），并返回对应特征切片。
    注意：node_subgraph 会重排节点ID为 [0..N0-1]，原ID保存在 G_oo.ndata[dgl.NID]。
    """
    dev = G_adv.device
    keep = torch.arange(N0, device=dev)
    G_oo = dgl.node_subgraph(G_adv, keep)        # 只含原始节点与其间的边
    X_oo = X_adv[:N0].clone()                    # 原始节点特征部分
    return G_oo, X_oo

def _edges_to_set(G: dgl.DGLGraph, undirected: bool):
    u, v = G.edges()
    uu, vv = u.detach().cpu().tolist(), v.detach().cpu().tolist()
    if undirected:
        return set((min(a,b), max(a,b)) for a,b in zip(uu, vv))
    else:
        return set((a, b) for a,b in zip(uu, vv))

def compare_graphs_ref_vs_advOO(
    G_ref: dgl.DGLGraph, X_ref: torch.Tensor,
    G_advOO: dgl.DGLGraph, X_advOO: torch.Tensor,
    undirected: bool = False, rtol=0.0, atol=0.0
):
    """
    对比“参考干净图(全原始节点)” vs “从 G_adv 索引出的原始部分(advOO)”：
    - 检查边集合是否完全一致（原始-原始之间不能被改动）
    - 检查原始节点特征是否一致（允许 rtol/atol 容差）
    """
    # 1) 边集合
    E_ref   = _edges_to_set(G_ref, undirected)
    E_advOO = _edges_to_set(G_advOO, undirected)
    removed = E_ref - E_advOO
    added   = E_advOO - E_ref
    edges_ok = (len(removed) == 0 and len(added) == 0)
    if edges_ok:
        print("[VERIFY][Edges] 原始-原始(OO) 边完全一致。")
    else:
        print(f"[VERIFY][Edges] 发现改动：被删={len(removed)}, 被加={len(added)}")
        if removed: print("  删掉的示例：", list(removed)[:10])
        if added:   print("  新增的示例：", list(added)[:10])

    # 2) 特征
    if X_ref.shape != X_advOO.shape:
        print(f"[VERIFY][X] 形状不一致: ref={tuple(X_ref.shape)}, advOO={tuple(X_advOO.shape)}")
        feats_ok = False
    else:
        diff = (X_ref - X_advOO).abs()
        feats_ok = bool(torch.all(diff <= (atol + rtol * X_ref.abs())).item())
        if feats_ok:
            print("[VERIFY][X] 原始节点特征一致（未被扰动）。")
        else:
            changed_rows = (diff.sum(dim=1) > 0).nonzero(as_tuple=True)[0].tolist()
            print(f"[VERIFY][X] 有 {len(changed_rows)} 个原始节点的特征发生改变，示例行：{changed_rows[:10]}")

    ok = edges_ok and feats_ok
    print(f"[VERIFY] 结论：{'通过' if ok else '不通过'}")
    return ok

def predict_labels_1d(victim, G, feats) -> torch.Tensor:
    with torch.no_grad():
        out = victim.predict(G, feats)
    if out.dim() == 2:           # [N, C] 浮点分数
        return out.argmax(dim=1)
    elif out.dim() == 1:         # [N] 整型标签
        return out.to(torch.long)
    else:
        raise ValueError(f"predict() output shape not supported: {tuple(out.shape)}")


def build_labels_with_injected(G, labels, anchor_to_injid, device):
    """
    基于原始 labels（长度=N_old），构造长度=N_current 的全图标签：
    - 原图节点：保持原始 labels
    - 恶性节点：标签 = 其对应锚点的标签

    参数
    ----
    G : dgl.DGLGraph          # 已经完成注入后的图（Stage-2b 之后）
    labels : torch.Tensor     # 原始图的真实标签，形状 [N_old]
    anchor_to_injid : dict    # {anchor_id: inj_id}
    device : torch.device

    返回
    ----
    labels_full : torch.Tensor  # 形状 [N_current]，恶性节点标签已赋值
    """
    N_old = int(labels.size(0))
    N_cur = int(G.num_nodes())
    labels_full = torch.full((N_cur,), -1, dtype=labels.dtype, device=device)

    # 1) 先写回原图节点的真实标签
    labels_full[:N_old] = labels.to(device)

    # 2) 给每个恶性节点赋“与锚点相同”的标签
    for a, inj in anchor_to_injid.items():
        a = int(a); inj = int(inj)
        labels_full[inj] = labels_full[a]  # 等同于 labels[a]，且在 device 上

    return labels_full

def build_args():
    p = argparse.ArgumentParser()
    # 数据/受害模型
    p.add_argument('--dataset', type=str, default='chameleon', choices=['cora','citeseer','pubmed','chameleon'])
    p.add_argument('--victim', type=str, default='gcn')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--hidden', type=int, default=8)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--feat_dropout', type=float, default=0.6)
    p.add_argument('--attn_dropout', type=float, default=0.6)
    p.add_argument('--num_heads1', type=int, default=8)
    p.add_argument('--num_heads2', type=int, default=1)
    p.add_argument('--sage_agg_type', type=str, default='mean')
    p.add_argument('--appnp_hidden', type=int, default=64)
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gin_hidden', type=int, default=64)
    p.add_argument('--gin_agg_type', type=str, default='sum')
    # 攻击目标
    p.add_argument('--attack_objective', type=str, default='untargeted',
                   choices=['untargeted','targeted'])
    p.add_argument('--target_ids', type=str, default='')  # 逗号分隔

    # 预算与稀疏
    # p.add_argument('--inj_ratio', type=float, default=0.1)  # 1%~10%
    p.add_argument('--deg_per_node', type=str, default='auto')  # 'auto' 或者整数
    p.add_argument('--max_rounds', type=int, default=1)
    # p.add_argument("--safe_k", type=int, default=2)
    # p.add_argument("--acc_eps", type=float, default=0.01)
    # p.add_argument("--kl_tau", type=float, default=0.05)
    # p.add_argument("--pinj_phase1", type=float, default=0.4)
    # p.add_argument("--topk_subgraph_edges", type=int, default=None)  # 可选
    # 超参可调
    p.add_argument('--opt_steps', type=int, default=200)
    p.add_argument('--opt_lr', type=float, default=0.1)
    p.add_argument('--hid_sur', type=int, default=64)
    # p.add_argument('--lam_kl', type=float, default=0.5)
    p.add_argument('--lam_l1', type=float, default=1e-3)
    # p.add_argument('--lam_budget', type=float, default=1.0)

    # 采样（有效电阻/谱稀疏化）
    # p.add_argument('--sampler', type=str, default='er', choices=['er'])
    # p.add_argument('--er_hutch_K', type=int, default=64)

    # 黑盒优化（NES/对比/正则）
    p.add_argument('--nes_edge_budget', type=int, default=8,
                   help='每个锚点最多新增的外边数；None 表示不做硬预算')
    p.add_argument('--nes_hard_topk', action='store_true',default=True,
                   help='在采样和落地时用 top-k 严格卡住边数（强烈建议打开）')
    p.add_argument('--nes_edge_tau', type=float, default=1.0,
                   help='边 logits 的温度（>1 更平缓，<1 更尖锐）')
    p.add_argument('--nes_edge_thr', type=float, default=0.5,
                   help='阈值二值化时的阈值（配合非 hard_topk 时用）')
    p.add_argument('--bb_queries', type=int, default=24)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--sigma', type=float, default=0.25)
    p.add_argument('--adam_lr', type=float, default=0.05)
    p.add_argument('--tau', type=float, default=0.5)     # 对比温度
    p.add_argument('--mmd_w', type=float, default=1.0)
    p.add_argument('--degkl_w', type=float, default=0.0)
    p.add_argument('--causal_w', type=float, default=1.0)
    p.add_argument('--l0_w', type=float, default=1e-3)
    p.add_argument('--khop_edge', type=int, default=2,
                        help='围绕锚点进行子图采样的 hop 数；0 表示不用采样，靠 1/2-hop 近邻集')
    p.add_argument('--p_stru', type=float, default=0.8,
                   help='AugmStru 的 Bernoulli 边保留概率（0~1，越大越贴近原图）')
    p.add_argument('--use_pseudo', action='store_true',
                   help='用于 AugmStru 的同类判定是否使用受害模型伪标签补全')
    # 剪边（隐蔽性）
    p.add_argument('--prune_ratio', type=float, default=0.3)

    # 训练/日志
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument("--lam_in", default=0.08, type=float)
    p.add_argument("--lam_out", default=0.1, type=float)
    p.add_argument("--lam_global", default=0.1, type=float)
    p.add_argument("--alpha_intra", default=0.0, type=float, help="0 表示自动用 CCDF 拟合；>0 则固定幂律指数")
    p.add_argument('--lam_edgecls', type=float, default=0.8,
                   help='子图稀疏优化中：优先连向目标类的边约束权重')

    #new

    # ===== NES 稳定性 / 退火 =====
    # p.add_argument('--dir_orthogonal', action='store_true',
    #                help='对随机方向做批内正交，提高黑盒梯度信噪比')
    # p.add_argument('--sigma_init', type=float, default=0.20,
    #                help='NES 噪声初值，用于退火')
    # p.add_argument('--sigma_final', type=float, default=0.08,
    #                help='NES 噪声末值，用于退火')
    # p.add_argument('--anneal_steps', type=int, default=40,
    #                help='前 anneal_steps 线性退火 sigma，之后恒定')
    p.add_argument('--log_every', type=int, default=10)

    # ===== 损失形态 =====
    p.add_argument('--loss_type', type=str, default='margin', choices=['margin', 'ce'],
                   help='anchor 分类项：margin(推荐) 或 ce')
    p.add_argument('--lam_class', type=float, default=1.2,
                   help='注入特征与目标类原型对齐的权重')

    # ===== 落地阶段：加-删-换 贪心 =====
    # p.add_argument('--enable_addrem', action='store_true',
    #                help='把新增和删除动作统一打分，只保留贡献最大的 k 条')
    # p.add_argument('--land_eval_topL', type=int, default=16,
    #                help='每个锚点仅评估概率最高的前 L 个新增候选，节省查询')
    # p.add_argument('--del_domain', type=str, default='m2m', choices=['m2m', 'all_out', 'none'],
    #                help='删除动作的边域：m2m=只删恶性↔恶性；all_out=删 inj 的任意外出；none=不删')
    # p.add_argument('--del_budget', type=int, default=None,
    #                help='每锚点最多可删多少条；默认与 nes_edge_budget 同步')
    # p.add_argument('--treat_undirected', action='store_true',
    #                help='把图按无向处理检查/删边（双向一致）')

    # ===== 落地期的 top-k 预算（不再覆盖为 avg_deg/2）=====
    # p.add_argument('--deg_per_node_landing', type=int, default=6,
    #                help='未显式给 nes_edge_budget 时，用这个估算每锚点新增边预算')
    # p.add_argument('--intra_ratio_landing', type=float, default=0.3,
    #                help='预算近似比例，和上面一起决定落地期 k_budget')

    # ===== SCM 因果先验 & 早停 =====
    # p.add_argument('--scm_guided', action='store_true',
    #                help='启用黑盒因果先验以初始化/引导 NES（强烈建议开启）')
    # p.add_argument('--scm_bias', type=float, default=1.2,
    #                help='因果先验注入到边 logits 的偏置强度')
    # p.add_argument('--scm_prior_L', type=int, default=16,
    #                help='每个锚点用于因果先验评估的候选数（随机抽样），越大越准但更慢')
    #
    # p.add_argument('--lam_smooth', type=float, default=0.0,
    #                help='特征-邻居平滑；>0 时注入特征与新增邻居的均值对齐')
    #
    # p.add_argument('--stop_on_hit', action='store_true',
    #                help='某个锚点达到目标类时立刻早停，切换下一个锚点')
    # p.add_argument('--early_patience', type=int, default=3,
    #                help='若最近多次监控无改进则早停该锚点')
    # p.add_argument('--restarts', type=int, default=1,
    #                help='NES 多重随机重启次数')
    # p.add_argument('--edge_warmup', type=int, default=0,
    #                help='只优化边的 warm-up 轮数')
    # p.add_argument('--feat_warmup', type=int, default=0,
    #                help='只优化特征的 warm-up 轮数')
    # p.add_argument('--k_expand', type=float, default=1.0,
    #                help='采样阶段放宽的边预算倍数（落地仍严格按预算）')
    p.add_argument('--inj_node_budget', type=float, default=0.01,
                   help='采样阶段放宽的边预算倍数（落地仍严格按预算）')
    return p.parse_args()


def k_hop_nodes_from_pair(adj_bin: torch.Tensor, src: int, dst: int, k: int):
    """
    adj_bin: (N,N) 0/1 稀疏 or 稠密张量（torch.float/torch.bool 都可）
    返回：子图节点下标（1D LongTensor，去重）
    """
    N = adj_bin.size(0)
    dev = adj_bin.device
    # 邻接布尔化
    A = (adj_bin > 0).to(torch.bool)
    seeds = torch.zeros(N, dtype=torch.bool, device=dev)
    seeds[src] = True; seeds[dst] = True
    frontier = seeds.clone()
    visited = seeds.clone()
    for _ in range(int(k)):
        # 无向扩展：in 或 out 都并上
        neib_out = (A & frontier.view(1, -1)).any(dim=1)   # 有出边到 frontier 的点
        neib_in  = (A & frontier.view(-1, 1)).any(dim=0)   # 有入边自 frontier 的点
        nbrs = neib_out | neib_in | frontier
        new_frontier = nbrs & (~visited)
        visited = visited | new_frontier
        frontier = new_frontier
        if not frontier.any(): break
    return visited.nonzero(as_tuple=True)[0].long()

def build_in_out_masks(N: int, sub_idx: torch.Tensor, device):
    """
    生成“入度视图/出度视图”的遮罩矩阵：
    - 入度视图：保留所有“指向 sub_idx 节点”的边 => mask_in[:, sub_idx] = 1
    - 出度视图：保留所有“由 sub_idx 节点发出”的边 => mask_out[sub_idx, :] = 1
    """
    mask_in  = torch.zeros((N, N), dtype=torch.float32, device=device)
    mask_out = torch.zeros((N, N), dtype=torch.float32, device=device)
    mask_in[:,  sub_idx] = 1.0
    mask_out[sub_idx, :] = 1.0
    return mask_in, mask_out

def clamp01(x):  # 与 SAVAGE 一致的 [0,1] 投影
    return torch.clamp(x, 0.0, 1.0)

def bce_link_loss(model, X, A, src, dst, target, device):
    """
    适配你现有 SAVAGE 的链路预测：decode(z, [[dst],[src]])
    target: 标量 0/1 (float)
    """
    z = model(X, A)  # (1,N,dim) or (N,dim)
    out = model.decode(z.squeeze(0), torch.tensor([[dst],[src]], device=device).long()).view(-1)
    crit = torch.nn.BCEWithLogitsLoss()
    tgt = torch.tensor([float(target)], device=device, dtype=torch.float32)
    return crit(out, tgt), torch.sigmoid(out).detach().item()

# 若你是“节点分类”任务，把上面函数换成：
# def ce_node_loss(model, X, A, y, node_idx, target_labels):
#     logits = model(X, A)  # (N,C)
#     return F.cross_entropy(logits[node_idx], target_labels[node_idx]), logits[node_idx].softmax(-1).detach()



def _parse_target_ids(s):
    if not s: return None
    try:
        ids = [int(t.strip()) for t in s.split(',') if t.strip()!='']
        return sorted(set(ids))
    except:
        return None
def feature_range_norm(feats: torch.Tensor, method: str = "maxabs"):
    """
    计算 'Feature range (norm)'。
    feats: [N, F] 的 torch.Tensor（可在 GPU）
    method: "maxabs"（默认，按列 |x| 的最大值做缩放），
            也可选 "zscore" 或 "minmax"（见注释）。
    返回:
        feats_norm: 归一化后的特征 (同 dtype/device)
        fmin, fmax: 归一化后在全图上的最小/最大值（float）
    """
    assert feats.dim() == 2, "feats should be [N, F]"
    X = feats

    if method == "maxabs":              # 每一列除以该列的 max(|x|)
        scale = X.abs().max(dim=0).values.clamp(min=1e-12)
        Xn = X / scale
    elif method == "zscore":            # 每列 (x-mean)/std，然后统计整体 min/max
        mu  = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False).clamp(min=1e-12)
        Xn = (X - mu) / std
    elif method == "minmax":            # 每列线性映射到 [-1, 1]
        xmin = X.min(dim=0).values
        xmax = X.max(dim=0).values
        denom = (xmax - xmin).clamp(min=1e-12)
        Xn = (X - xmin) / denom        # -> [0,1]
        Xn = Xn * 2 - 1                 # -> [-1,1]
    else:
        raise ValueError(f"unknown method: {method}")

    # 统计归一化后的全局范围
    fmin = float(Xn.min().detach().cpu())
    fmax = float(Xn.max().detach().cpu())
    return Xn, fmin, fmax
def main():
    args = build_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1) 加载数据与受害模型
    # G, feats, labels, train_idx, val_idx, test_idx = load_dgl_dataset(args.dataset, device)
    G, feats, labels, train_idx, val_idx, test_idx = load_original_graph(args.dataset, device, args)
    feats_norm, fmin, fmax = feature_range_norm(feats, method="minmax")
    print(f"[Feature range (norm)] {fmin:.2f} ~ {fmax:.2f}")
    print(f"[Feature num: {feats.shape[1]}]")

    C = int(labels.max().item() + 1)
    print(f"Class num: {C}")
    victim = Victim(model_name=args.victim, in_dim=feats.shape[1],
                    n_classes=int(labels.max().item()+1),
                    lr=args.lr, wd=args.wd, args=args, device=device)

    print("\n[Stage 0] Train victim on clean graph…")
    print('注入的是:{}网络'.format(args.victim))
    clean = victim.fit_eval(G, feats, labels, train_idx, val_idx, test_idx,
                            epochs=args.epochs, patience=args.patience, verbose=True)
    print(f"Clean test acc: {clean['test_acc']*100:.2f}%")
    print("注入了{}比例的节点".format(args.inj_node_budget))
    # 2) 因果打分 & 锚点
    cscorer = CausalScorer(victim=victim, device=device, alpha=0.6, drop_top=5)
    print("\n[Stage 1] Scoring node vulnerability (causal proxy)…")
    vul_scores = cscorer.score_vulnerability(G, feats, labels, test_idx, topk= min(G.num_nodes()*args.inj_node_budget, int(test_idx.shape[0])))
    anchors = vul_scores['top_nodes']  # 高易损节点做 anchor
    print(f"Picked {len(anchors)} anchors as vulnerable nodes.")
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()

    G_clean_snapshot = G.clone()
    X_clean_snapshot = feats.clone()

    # 3) 注入规模/度预算
    N0 = G.num_nodes()
    # inj_nodes = max(1, int(round(N0 * args.inj_ratio)))
    inj_nodes = len(anchors)  # ← 每个子图 1 个注入节点（原来是按 inj_ratio）
    avg_deg = int(round(float(G.num_edges())/N0))*2
    # args.nes_edge_budget = avg_deg/2
    print("平均度为:{}".format(avg_deg))
    print("边预算为{}".format(args.nes_edge_budget))
    deg_per_node = avg_deg if args.deg_per_node=='auto' else int(args.deg_per_node)
    total_new_edges = inj_nodes * deg_per_node
    print(f"[Budget] inject nodes = ({args.inj_node_budget*100:.1f}%), "
          f"deg per injected = {deg_per_node}, total new edges ≈ {total_new_edges}")
    # 如果命令行没给预算，就用一个小而稳的默认
    if args.nes_edge_budget is None:
        args.nes_edge_budget = max(2, int(0.3 * deg_per_node))  # 例如 0.3*deg_per_node
    print(f"  [Original] graph size N={base_nodes}, E={base_edges} (no structural change)")

    print("\n[Stage 2] Building ER-guided candidate pool…")
    base_pool = get_khop_candidates(
        G=G,
        anchors=anchors,  # Stage 1 得到的脆弱锚点（tensor）
        k_hop_edge=args.khop_edge,  # 例如 1 或 2；0 表示用全图的 1/2-hop 近邻集
        deg_per_node=int(round(0.7*deg_per_node)),  # 你的预算：每个注入点要连多少条边
        fallback_random=True
    )
    print("  Candidate pool ready (k-hop).")
    print_khop_subgraph_stats(G, anchors, k_hop_edge=args.khop_edge)
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()
    print(f"  [Stage-1 done] graph size N={base_nodes}, E={base_edges} (no structural change)")

    if args.use_pseudo:
        with torch.no_grad():
            pseudo_logits = victim.predict(G, feats)
        labels_for_sampler = pseudo_logits  # 关键：取 argmax 得到 1D 标签
    else:
        labels_for_sampler = labels
    # weights = dict(homo=1.0, degree=0.3, cos=0.5)  # 起步比较稳的权重
    # weights = dict(homo=1.6, degree=0.1, cos=0.8)
    weights = dict(homo=1, degree=1, cos=1)
    candidate = augmstru_enhanced_bernoulli(
        G=G,
        anchors=anchors,
        base_pool=base_pool,
        # K=int(round(0.7*deg_per_node)),
        K=int(round(0.9*deg_per_node)),
        labels=labels_for_sampler,
        feats=feats,
        weights=weights,
        # temp=1.0,
        temp=0.7,
        exact_k=True
    )
    G, inj_ids, anchor_to_injid = stage2b_inject_per_subgraph(
        G=G,
        anchors=anchors,
        candidate=candidate,
        deg_per_node=int(round(0.7*deg_per_node)),
        feats=feats,  # 若 G.ndata['feat'] 已有也可不传
        undirected=True,
        verbose=True
    )
    # 同步外部 feats 引用（后续统一用 G.ndata['feat'] 更稳）
    feats = G.ndata['feat']
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()
    print(f"[After Stage 2b] N={G.num_nodes()}, E={G.num_edges()}")
    print("  Candidate pool refined by Enhanced-Bernoulli (homo/degree/cos mixed).")

    per_node_inner_used = None
    # print(inj_ids,candidate)
    labels_full = build_labels_with_injected(G, labels, anchor_to_injid, device)
    G_before_E = base_edges
    # 如果后面需要“误导视图”的标签（labels_for_sampler）以此为起点：
    labels_for_True = labels_full.clone()
    G, inj_pairs = stage3_connect_between_injected(
        G=G,
        inj_ids=inj_ids,
        deg_per_node=deg_per_node,
        intra_ratio=0.3,
        per_node_inner_used=per_node_inner_used,
        undirected=False,  # ← 可写可不写，已忽略
        seed=42,
        verbose=True,
        # alpha_intra=args.alpha_intra,  # 0=自动拟合；>0=固定指数
        alpha_intra=args.alpha_intra if args.alpha_intra > 0 else 2.0,
        tgt_pref="indegree",  # 或 "uniform"
        # tgt_gamma=1.0
        tgt_gamma=1.5
    )

    print(f"[After Stage 3] N={G.num_nodes()}, E={G.num_edges()} (+{G.num_edges() - G_before_E} edges)")
    use_pseudo_after3 = bool(args.use_pseudo)  # 布尔就用布尔，不和字符串比较
    device = feats.device
    with torch.no_grad():
        logits = victim.predict(G, feats)  # [N, C] 或 [N]，兼容性处理
        print(logits.dim())
    if logits.dim() == 1:
        # 如果模型只给 [N]（极少数情况），退化为随机非真类
        y_pred = logits.to(torch.long)
        C = int(labels.max().item() + 1)
        gen = torch.Generator(device=feats.device).manual_seed(getattr(args, "seed", 0))
        r = torch.randint(low=0, high=C - 1, size=y_pred.shape, generator=gen, device=feats.device)
        labels_for_sampler = (r + (r >= y_pred)).to(torch.long)
    else:
        y_pred = logits.argmax(dim=1)
        # 目标 = 第二大概率类（比“最小概率类”更容易打过去）
        top2 = logits.topk(2, dim=1).indices
        y_tgt_sec = top2[:, 1]
        # 保险：极少情况下 top2[:,1] 可能与 argmax 相同（数值并列），兜底换成“最小概率类”
        same = (y_tgt_sec == y_pred)
        if same.any():
            y_min = logits.argmin(dim=1)
            y_tgt_sec = torch.where(same, y_min, y_tgt_sec)
        labels_for_sampler = y_tgt_sec.to(torch.long)

    labels_for_sampler = labels_for_sampler.to(device=feats.device, dtype=torch.long)


    # 类原型（特征对齐用）
    protos = []
    for c in range(C):
        idx = (labels_for_sampler == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            protos.append(torch.zeros(feats.size(1), device=device))
        else:
            protos.append(feats[idx].mean(dim=0))
    class_protos = torch.stack(protos, dim=0).to(device)

    print(f"[After Stage 3] labels_for_sampler ready (mode={use_pseudo_after3}).")

    # ===== 2) 锚点-注入映射合并 & 子图抽取（保留你原来的打印，做健壮处理）=====
    merged_by_anchor, injid_to_cands, edge_pairs = merge_anchor_maps(anchor_to_injid, base_pool)
    # print(merged_by_anchor)
    # sample_anchor = 2323
    # if sample_anchor in merged_by_anchor:
    #     print(merged_by_anchor[sample_anchor])
    # else:
    #     print(f"[Info] anchor {sample_anchor} not in merged_by_anchor; show first:",
    #           next(iter(merged_by_anchor.items())))
    # sample_inj = 2708
    # if sample_inj in injid_to_cands:
    #     print(injid_to_cands[sample_inj])
    # else:
    #     print(f"[Info] inj {sample_inj} not in injid_to_cands; show first:", next(iter(injid_to_cands.items())))
    print(f"共提取子图数量: {len(merged_by_anchor)}")

    result = batch_subgraphs_from_injid_to_cands(G, injid_to_cands)
    # probe_inj = sample_inj if sample_inj in result else (next(iter(result)) if len(result) > 0 else None)
    # if probe_inj is not None:
    #     sg, nodes, mapping = result[probe_inj]
    #     print(f"[inj {probe_inj}] subgraph |V|={sg.num_nodes()}, |E|={sg.num_edges()}")
    #     print("  nodes(list in G):", nodes.tolist())
    #     print("  subg_node -> global_id:", mapping.tolist())
    # else:
    #     print("[Warn] 没有可用的 inj 子图（injid_to_cands 为空？）")
    G_before_Edge = G.num_edges()

    with torch.no_grad():
        # logits_final = victim.predict(G, feats)
        y_hat = predict_labels_1d(victim, G, feats)
    # y_hat = logits_final.argmax(dim=1)
    gt = labels_for_True.to(y_hat.device)

    # （1）测试集准确率（节点分类）
    if 'test_idx' in locals() and test_idx is not None:
        test_acc_final = (y_hat[test_idx] == gt[test_idx]).float().mean().item()
        print(f"[Final] Test accuracy (GT labels): {test_acc_final * 100:.2f}%")
    from Subgraph_Sparse_Attack import run_subgraph_sparse_attack_nes

    G_adv, feats_adv = run_subgraph_sparse_attack_nes(
        G=G,
        feats=feats,
        victim=victim,
        merged_by_anchor=merged_by_anchor,
        labels_for_sampler=labels_for_sampler,
        class_protos=class_protos,
        args=args,
        T_homo=0.7,
        log_every=args.log_every
    )

    print(f"[After SubgraphSparseAttack] N={G_adv.num_nodes()}, E={G_adv.num_edges()}")
    G_adv00, X_adv00 = extract_clean_part_from_adv(G_adv, feats_adv, N0)
    _ = compare_graphs_ref_vs_advOO(
        G_ref=G_clean_snapshot, X_ref=X_clean_snapshot,
        G_advOO=G_adv00, X_advOO=X_adv00,
        undirected=False  # 如果你的图逻辑上当作无向，这里设 True
    )
    # 使用对抗图/特征
    G, feats = G_adv, feats_adv

    # ===== 4) 最终评测（务必用真实标签 labels）=====
    with torch.no_grad():
        # logits_final = victim.predict(G, feats)
        y_hat = predict_labels_1d(victim, G, feats)
    # y_hat = logits_final.argmax(dim=1)
    gt = labels_for_True.to(y_hat.device)

    # （1）测试集准确率（节点分类）
    if 'test_idx' in locals() and test_idx is not None:
        test_acc_final = (y_hat[test_idx] == gt[test_idx]).float().mean().item()
        print(f"[Final] Test accuracy (GT labels): {test_acc_final * 100:.2f}%")

    # （2）全图准确率（可选）
    overall_acc_final = (y_hat == gt).float().mean().item()
    print(f"[Final] Overall accuracy (GT labels): {overall_acc_final * 100:.2f}%")

    # （3）锚点定向成功率：预测==误导标签（分析指标）
    asr = (y_hat[anchors] != labels_for_True[anchors].to(y_hat.device)).float().mean().item()
    print(f"[Analysis] Targeted success on anchors (toward misguide labels): {asr * 100:.2f}%")

    # （4）结构改动量（相对 Stage-3 前）
    print(f"[Delta] Added edges after sparse attack (approx): {G.num_edges() - G_before_Edge}")
    print(G.num_edges())
    print(G.num_nodes())
    try:
        print("\n[Defense Eval] Evaluating defenses on the adversarial graph (CPU mode)...")

        # 1) 保留一份 GPU 版本不动（后面你的 victim 评测还可以继续用 GPU）
        G_adv_gpu, feats_adv_gpu = G_adv, feats_adv

        # 2) 拷贝一份到 CPU 专供防御评测
        G_adv_cpu = G_adv_gpu.to('cpu')  # DGL 图迁移到 CPU
        feats_cpu = feats_adv_gpu.to('cpu')  # 特征到 CPU
        labels_cpu = labels.to('cpu')  # 真实标签到 CPU
        tr_cpu = train_idx.to('cpu')
        va_cpu = val_idx.to('cpu')
        te_cpu = test_idx.to('cpu')

        # 3) 调 evaluate_defenses，明确 device=cpu
        from defense_eval import evaluate_defenses
        defense_results = evaluate_defenses(
            G_adv_cpu, feats_cpu, labels_cpu,
            tr_cpu, va_cpu, te_cpu,
            dataset=args.dataset,
            model_names=("gnnguard", "egnnguard", "rgat"),  # 精简下别名也行
            hidden=128, layers=3, dropout=0.5,
            use_ln=0, ln_first=False, homo_threshold=0.1,
            device=torch.device('cpu'),  # 关键：CPU 设备
            epochs=300, lr=0.01, wd=0.0, patience=50,
        )

        # 4) 评测完，把对抗图/特征继续用原来的 GPU 变量
        G_adv, feats_adv = G_adv_gpu, feats_adv_gpu
        # 如需访问结果字典：
        # print(defense_results)
    except Exception as e:
        print("[Defense Eval] Skipped due to error:", repr(e))


if __name__ == '__main__':
    main()
