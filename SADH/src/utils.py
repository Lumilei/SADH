import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

# =========================
# 节点初始化与特征填补
# Node initialization and feature imputation
# =========================
def l2_normalize_features(X):
    """
    对特征矩阵进行L2归一化，先用均值填充NaN/Inf，剩余NaN再填0
    """
    X = np.where(np.isinf(X), np.nan, X)
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    # 再次检查，若还有NaN（如全列NaN），填0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize(X, norm='l2', axis=1)

def compute_view_centroid(X, mask):
    """
    计算视图质心（所有可用样本的均值）
    Compute view centroid (mean of all available samples)
    """
    valid = mask.astype(bool)
    if np.sum(valid) == 0:
        return np.zeros(X.shape[1])
    return np.mean(X[valid], axis=0)

def feature_imputation(X, mask, view_centroid):
    """
    用视图质心填补缺失特征
    Impute missing features with view centroid
    """
    X_filled = X.copy()
    for i in range(X.shape[0]):
        if not mask[i]:
            X_filled[i] = view_centroid
    return X_filled

def initialize_sample_embedding(X_views, mask_row):
    """
    样本节点初始化：对所有可用视图特征取均值
    Initialize sample node embedding: mean of all available view features
    """
    valid_views = [X for X, m in zip(X_views, mask_row) if m]
    if len(valid_views) == 0:
        return np.zeros(X_views[0].shape)
    return np.mean(valid_views, axis=0)

# =========================
# 置信度与冲突建模
# Confidence and conflict modeling
# =========================
def initial_confidence(x_k, x_others, r_i):
    """
    初始置信度计算
    Initial confidence calculation
    """
    if len(x_others) == 0:
        return 0.5 * (1 - r_i)
    cos_sim = np.mean([np.dot(x_k, x_l) / (np.linalg.norm(x_k) * np.linalg.norm(x_l) + 1e-8) for x_l in x_others])
    return 1 / (1 + np.exp(-cos_sim * (1 - r_i)))

def dynamic_confidence(h_k, h_others, r_i):
    """
    动态置信度计算
    Dynamic confidence calculation
    """
    if len(h_others) == 0:
        return 0.5 * (1 - r_i)
    cos_sim = np.mean([np.dot(h_k, h_l) / (np.linalg.norm(h_k) * np.linalg.norm(h_l) + 1e-8) for h_l in h_others])
    return 1 / (1 + np.exp(-cos_sim * (1 - r_i)))

def conflict_js(p_k, p_bar, alpha, semantic_conflict):
    """
    视图冲突度建模
    View conflict modeling
    """
    from scipy.spatial.distance import jensenshannon
    js = jensenshannon(p_k, p_bar)
    return alpha * js + (1 - alpha) * semantic_conflict

# =========================
# SMOTE增强
# SMOTE augmentation
# =========================
def smote_augmentation(X, y, random_state=42):
    """
    对少数类进行SMOTE合成增强，自动调整k_neighbors，类别样本数<2或k_neighbors<1时跳过SMOTE
    """
    from collections import Counter
    class_counts = Counter(y)
    if len(class_counts) < 2:
        print("SMOTE跳过：训练集类别数<2")
        return X, y
    min_count = min(class_counts.values())
    if min_count < 2:
        print("SMOTE跳过：存在样本数<2的类别")
        return X, y
    k_neighbors = min(5, min_count - 1)
    if k_neighbors < 1:
        print("SMOTE跳过：k_neighbors<1")
        return X, y
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# =========================
# 对比损失
# Contrastive loss
# =========================
def contrastive_loss(embeddings, labels, temperature=0.5):
    """
    计算NT-Xent对比损失（SimCLR风格）
    Compute NT-Xent contrastive loss (SimCLR style)
    embeddings: [N, d]，样本嵌入
    labels: [N]，类别标签
    temperature: 温度参数
    """
    device = embeddings.device
    N = embeddings.shape[0]
    # 归一化
    emb_norm = F.normalize(embeddings, dim=1)
    # 相似度矩阵
    sim_matrix = torch.matmul(emb_norm, emb_norm.T) / temperature  # [N, N]
    # 构造正样本掩码（同类为正，其余为负）
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # 排除自身
    self_mask = torch.eye(N, device=device)
    mask = mask - self_mask
    # 对每个样本，正样本对的分子，负样本对的分母
    exp_sim = torch.exp(sim_matrix) * (1 - self_mask)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    # 只对正样本对求平均
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss