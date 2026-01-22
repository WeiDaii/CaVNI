import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from defenses import EGCNGuard, GCNGuard, RobustGCN, RGAT
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Dict, Tuple

# ---------------------------
# Utils
# ---------------------------

def dgl_to_sparsetensor(g_dgl) -> SparseTensor:
    """Convert a DGLGraph to torch_sparse.SparseTensor with unit weights (later reweighted by defenses)."""
    src, dst = g_dgl.edges()
    n = g_dgl.num_nodes()
    row = src.long().cpu()
    col = dst.long().cpu()
    val = torch.ones(row.numel(), dtype=torch.float32)
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(n, n)).coalesce()

@torch.no_grad()
def _eval(model: torch.nn.Module, x: torch.Tensor, adj: SparseTensor, idx: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    out = model(x, adj)  # log_softmax
    pred = out[idx].argmax(dim=-1)
    return (pred == y[idx]).float().mean().item()

def _train(model: torch.nn.Module,
           x: torch.Tensor,
           adj: SparseTensor,
           train_idx: torch.Tensor,
           val_idx: torch.Tensor,
           y: torch.Tensor,
           epochs: int = 500,
           lr: float = 0.01,
           wd: float = 0.0,
           patience: int = 50) -> torch.nn.Module:
    """Standard supervised training loop with early stopping on val acc (run_cora style defaults)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, wait, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, adj)  # log_softmax
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        val = _eval(model, x, adj, val_idx, y)
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model

def align_labels_to_graph(G, labels) -> torch.Tensor:
    """Pad/trim labels to match graph nodes; injected nodes get -1."""
    N = G.num_nodes()
    lbl = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
    if lbl.numel() < N:
        pad = torch.full((N - lbl.numel(),), -1, dtype=torch.long)
        lbl = torch.cat([lbl, pad], dim=0)
    elif lbl.numel() > N:
        lbl = lbl[:N]
    return lbl

def build_indices(tr: torch.Tensor, va: torch.Tensor, te: torch.Tensor, labels: torch.Tensor):
    """Remove unlabeled (-1) nodes from splits (consistent with your pipeline)."""
    lbl = labels.view(-1)
    valid = (lbl >= 0)
    tr_idx = tr[valid[tr]]
    va_idx = va[valid[va]]
    te_idx = te[valid[te]]
    return tr_idx, va_idx, te_idx

# ---------------------------
# Model factory (mirrors gnn_misg.py & run_cora.sh)
# ---------------------------

def build_model(name: str,
                dataset: str,
                in_dim: int,
                hid: int,
                out_dim: int,
                layers: int,
                dropout: float,
                use_ln: bool,
                ln_first: bool,
                homo_threshold: float) -> torch.nn.Module:
    nm = name.lower()
    ds = (dataset or "grb-cora").lower()

    if nm == "egnnguard":
        # Threshold policy consistent with gnn_misg.py
        if ds in ["grb-reddit", "computers"]:
            th = 0.15
        elif ds in ["computers"]:
            th = 0.3
        else:
            th = homo_threshold
        return EGCNGuard(in_dim, hid, out_dim, layers, dropout,
                         layer_norm_first=ln_first, use_ln=use_ln, threshold=th)

    elif nm in ("gnnguardor", "gnnguard", "gnnguardwa"):
        # 'or' → original, no learnable attention drop; others enable it
        attention_drop = (nm != "gnnguardor")
        return GCNGuard(in_dim, hid, out_dim, layers, dropout,
                        layer_norm_first=ln_first, use_ln=use_ln, attention_drop=attention_drop)

    elif nm == "robustgcn":
        return RobustGCN(in_dim, hid, out_dim, layers, dropout)

    elif nm == "rgat":
        # RGAT special flags in your repo: disable att_dropout; att_cpu on reddit
        # These flags are handled inside RGAT's own __init__; here we keep the basic signature.
        return RGAT(in_dim, hid, out_dim, layers, dropout)

    else:
        raise ValueError(f"Unknown defense model: {name}")

# ---------------------------
# Defaults from run_cora.sh / gnn_misg.py
# ---------------------------

def _defaults_from_run_cora(dataset: str, model_name: str) -> dict:
    ds = (dataset or "grb-cora").lower()
    name = (model_name or "egnnguard").lower()
    cfg = dict(
        hidden=128,
        layers=3,
        dropout=0.5,
        use_ln=0,
        ln_first=False,
        epochs=500,
        lr=0.01,
        wd=0.0,
        patience=50,
        homo_threshold=0.1,
    )
    if "egnnguard" in name:
        if ds in ["grb-reddit", "computers"]:
            cfg["homo_threshold"] = 0.15
        elif ds in ["computers"]:
            cfg["homo_threshold"] = 0.3
        else:
            cfg["homo_threshold"] = 0.1
    if name == "rgat":
        cfg["use_ln"] = 1  # RGAT commonly with LN in your scripts
    return cfg

# ---------------------------
# Public API
# ---------------------------

def evaluate_defenses(G_adv, feats, labels, tr, va, te,
                      dataset: str = "grb-cora",
                      model_names=("gnnguardor", "robustgcn", "egnnguard"),
                      hidden: int = None, layers: int = None, dropout: float = None,
                      use_ln: int = None, ln_first: bool = None, homo_threshold: float = None,
                      device: str = None, epochs: int = None, lr: float = None,
                      wd: float = None, patience: int = None) -> Dict[str, float]:

    _cfg = _defaults_from_run_cora(dataset, model_names[0] if model_names else "egnnguard")
    hidden      = _cfg["hidden"] if hidden is None else hidden
    layers      = _cfg["layers"] if layers is None else layers
    dropout     = _cfg["dropout"] if dropout is None else dropout
    use_ln      = _cfg["use_ln"] if use_ln is None else use_ln
    ln_first    = _cfg["ln_first"] if ln_first is None else ln_first
    homo_threshold = _cfg["homo_threshold"] if homo_threshold is None else homo_threshold
    epochs      = _cfg["epochs"] if epochs is None else epochs
    lr          = _cfg["lr"] if lr is None else lr
    wd          = _cfg["wd"] if wd is None else wd
    patience    = _cfg["patience"] if patience is None else patience

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 特征/标签放到 device
    x = torch.as_tensor(feats, dtype=torch.float32, device=device)
    y = align_labels_to_graph(G_adv, labels).to(device)

    tr_idx, va_idx, te_idx = build_indices(tr, va, te, y.cpu())
    tr_idx, va_idx, te_idx = tr_idx.to(device), va_idx.to(device), te_idx.to(device)

    # 关键：邻接先在 CPU 构，再整体搬到 device（GPU）
    adj = dgl_to_sparsetensor(G_adv)
    if device != "cpu":
        adj = adj.to(device)

    results: Dict[str, float] = {}
    for name in model_names:
        print(f"[Defense] Training {name} on {device} …")

        model = build_model(name, dataset,
                            in_dim=x.size(1), hid=hidden, out_dim=int(y.max().item() + 1),
                            layers=layers, dropout=dropout,
                            use_ln=bool(use_ln), ln_first=bool(ln_first),
                            homo_threshold=homo_threshold).to(device)

        model = _train(model, x, adj, tr_idx, va_idx, y,
                       epochs=epochs, lr=lr, wd=wd, patience=patience)
        acc = _eval(model, x, adj, te_idx, y)
        results[name] = acc
        print(f"[Defense] {name}: test acc on adversarial graph = {acc * 100:.2f}%")
        print(f"[Defense] {name} on", next(model.parameters()).device)

    return results
