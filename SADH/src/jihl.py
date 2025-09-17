import torch
import torch.nn as nn
import torch.nn.functional as F

class JIHLImputer(nn.Module):
    def __init__(self, view_slices, feature_dim, hidden_dim=128):
        super().__init__()
        self.view_slices = view_slices
        self.mlp = nn.Sequential(
            nn.Linear(sum([s[1]-s[0] for s in view_slices])*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sum([s[1]-s[0] for s in view_slices]))
        )
        self.ema_alpha = 0.9

    def forward(self, X, mask, h_views, lowconf_edges, infotrans_edges):
        # X: [N, D], mask: [N, D], h_views: [N, M, d_h]
        N, D = X.shape
        M = len(self.view_slices)
        device = X.device

        # 1. 置信度和冲突度
        confidence = torch.zeros(N, M, device=device)
        conflict = torch.zeros(N, M, device=device)
        for i in range(N):
            for k, (start, end) in enumerate(self.view_slices):
                if mask[i, start:end].sum() == (end-start):  # 该视图不缺失
                    available = [l for l in range(M) if l != k and mask[i, self.view_slices[l][0]:self.view_slices[l][1]].sum() == (self.view_slices[l][1]-self.view_slices[l][0])]
                    if len(available) > 0:
                        h_k = h_views[i, k]
                        h_others = h_views[i, available]
                        cos_sim = F.cosine_similarity(h_k.unsqueeze(0), h_others, dim=1).mean()
                        confidence[i, k] = torch.sigmoid(cos_sim)
                        h_mean = h_others.mean(dim=0)
                        conflict[i, k] = (1 - F.cosine_similarity(h_k, h_mean, dim=0)) / 2
                    else:
                        confidence[i, k] = 0.5
                        conflict[i, k] = 0.0
                else:
                    confidence[i, k] = 0.5
                    conflict[i, k] = 0.0

        # 2. 动态填补
        X_hat = X.clone()
        for i in range(N):
            for k, (start, end) in enumerate(self.view_slices):
                if mask[i, start:end].sum() < (end-start):  # 该视图有缺失
                    # 跨视图信息
                    available = [l for l in range(M) if l != k and mask[i, self.view_slices[l][0]:self.view_slices[l][1]].sum() == (self.view_slices[l][1]-self.view_slices[l][0])]
                    if available:
                        alpha = F.softmax(confidence[i, available] * (1 - conflict[i, available]), dim=0)
                        info_view = sum(alpha[j] * X[i, self.view_slices[available[j]][0]:self.view_slices[available[j]][1]] for j in range(len(available)))
                    else:
                        info_view = torch.zeros(end-start, device=device)
                    # 同类困难样本信息（低置信度超边）
                    info_hard = torch.zeros(end-start, device=device)
                    # ...遍历lowconf_edges，聚合同视图难样本...
                    # 高置信度邻居信息（信息传递超边）
                    info_help = torch.zeros(end-start, device=device)
                    # ...遍历infotrans_edges，聚合同视图高置信度邻居...
                    # 拼接
                    concat = torch.cat([info_view, info_hard, info_help], dim=0)
                    pred = self.mlp(concat)
                    # EMA平滑
                    X_hat[i, start:end] = self.ema_alpha * X_hat[i, start:end] + (1 - self.ema_alpha) * pred
        return X_hat
