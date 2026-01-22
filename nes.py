# -*- coding: utf-8 -*-
"""
NES 黑盒优化 (Ilyas et al., ICML'18) + Adam 风格动量
离散边选择用 STE (Bengio et al., 2013) 做二值化/反传近似
"""
import torch

class NESAdam:
    def __init__(self, sigma=0.25, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, device='cpu'):
        self.sigma, self.lr = sigma, lr
        self.m, self.v = None, None
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.t = 0
        self.device = device

    def step(self, Z, f_eval, queries=64):
        """
        Z: 连续参数 (形如 [M, K] 每个注入节点的 K 个候选边得分)
        f_eval: callable(binary_edges_mask)->scalar  黑盒目标函数（越大越好/或越小越好）
        """
        self.t += 1
        M,K = Z.shape
        g = torch.zeros_like(Z)
        for _ in range(queries//2):
            u = torch.randn_like(Z)
            Zp, Zn = Z + self.sigma*u, Z - self.sigma*u
            # STE: 二值化 (top-k 由外侧控制；此处只做阈值01近似)
            Yp = torch.sigmoid(Zp); Yn = torch.sigmoid(Zn)
            fp = f_eval(Yp); fn = f_eval(Yn)
            g += ((fp - fn)/(2*self.sigma)) * u
        g /= (queries//2)

        # Adam
        if self.m is None: self.m = torch.zeros_like(Z); self.v = torch.zeros_like(Z)
        self.m = self.b1*self.m + (1-self.b1)*g
        self.v = self.b2*self.v + (1-self.b2)*(g*g)
        mhat = self.m/(1-self.b1**self.t)
        vhat = self.v/(1-self.b2**self.t)
        Z = Z + self.lr * mhat / (torch.sqrt(vhat)+self.eps)  # 上升
        return Z
