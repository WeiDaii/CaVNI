# CaVNI: Causal Vulnerability-Aware Sparse Node Injection Attacks against Black-Box Graph Neural Networks

This repository provides the implementation of **CaVNI**, a new framework for sparse node injection against black-box GNNs.  
CaVNI focuses on (i) vulnerable subgraph generation, (ii) sparse injection under strict budgets, and (iii) gradient-free refinement (NES-style) with limited black-box queries.

<p align="center"> <img src="fig/CaVNI.jgp" /> <p align="center"><em>Fig. 3: Overall framework of CaVNI. The framework consists of three modules: vulnerable subgraph generation, edge sampling and injection, and sparse optimization. In particular, steps (1)–(5) in the sparse optimization module depict the iterative workflow of each optimization cycle.</em></p>

## Highlights
- **Black-box setting**: only queryable forward outputs are required.
- **Vulnerable subgraph generation**: localizes a small set of vulnerable anchors and extracts k-hop subgraphs to reduce the search space.
- **Sparse optimization**: jointly refines injected edges/features with **NES** under strict budgets.
- **Strict-budget landing**: discretized/Top-k landing ensures the final injected structure obeys the given edge budget.

---

## Requirements

### Tested Environment
- OS: Ubuntu (GPU environment)
- CUDA: 11.8 (cu118)
- PyTorch: **2.1.2+cu118**
- DGL: **1.1.3+cu118**
- PyTorch Geometric: **2.6.1** (with matching CUDA wheels)

---

## Installation

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
