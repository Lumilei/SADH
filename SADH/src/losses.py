import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(embeddings, labels, tau=0.07):
    """对比损失"""
    embeddings = F.normalize(embeddings, dim=1)
    N = embeddings.size(0)
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / tau
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    logits_mask = torch.ones_like(mask) - torch.eye(N, device=embeddings.device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

def reconstruction_loss(X_pred, X_true, mask=None):
    """重建损失"""
    if mask is not None:
        loss = F.mse_loss(X_pred * mask, X_true * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-8)
    else:
        loss = F.mse_loss(X_pred, X_true)
    return loss

def compute_total_loss(out, x, y, config=None, sample_weights=None):
    """计算总损失，支持加权交叉熵"""
    if config is None:
        config = {
            'use_cls': True,
            'use_contrastive': False,
            'use_reconstruction': False,
            'lambda_con': 0.1,
            'lambda_recon': 0.1,
            'tau': 0.07
        }
    
    total_loss = 0.0
    loss_components = {}
    
    # 分类损失
    if config.get('use_cls', True):
        if sample_weights is not None:
            ce_loss = F.cross_entropy(out, y, reduction='none')
            loss_cls = (ce_loss * sample_weights).mean()
        else:
            loss_cls = F.cross_entropy(out, y)
        total_loss += loss_cls
        loss_components['cls'] = loss_cls.item()
    
    # 对比损失
    if config.get('use_contrastive', False):
        loss_con = contrastive_loss(x, y, tau=float(config['tau']))
        total_loss += float(config['lambda_con']) * loss_con
        loss_components['con'] = loss_con.item()
    
    # 重建损失（如果需要）
    if config.get('use_reconstruction', False):
        # 这里需要根据具体实现添加重建损失
        pass
    
    return total_loss, loss_components