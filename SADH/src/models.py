import torch
import torch.nn as nn
import torch.nn.functional as F
from hypergat import MultiLayerHyperGAT

class SimpleHypergraphConv(nn.Module):
    """简单超图卷积层"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, X, H):
        HX = torch.matmul(H.t(), X)
        D_e = H.sum(0, keepdim=True) + 1e-6
        HX = HX / D_e.t()
        X_new = torch.matmul(H, HX)
        return self.linear(X_new)

class SimpleESADH(nn.Module):
    """ESADH模型"""
    def __init__(self, in_dim, embed_dim, num_classes, dropout=0.5):
        super().__init__()
        self.hgconv1 = SimpleHypergraphConv(in_dim, embed_dim)
        self.relu = nn.ReLU()
        self.hgconv2 = SimpleHypergraphConv(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, X, H):
        x = self.hgconv1(X, H)
        x = self.relu(x)
        x = self.hgconv2(x, H)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out, x

class SimpleHyperGCN(nn.Module):
    """增强版超图GCN模型，集成HyperGAT和JIHL模块"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5, view_slices=None, view_dims=None, use_hypergat=True):
        super(SimpleHyperGCN, self).__init__()
        
        # 是否使用HyperGAT
        self.use_hypergat = use_hypergat
        
        if use_hypergat:
            # 使用HyperGAT层
            self.hypergat = MultiLayerHyperGAT(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=2,
                num_heads=8,
                dropout=dropout,
                alpha=0.2
            )
        else:
            # 使用简单的线性层（向后兼容）
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # 视图相关信息
        self.view_slices = view_slices
        self.view_dims = view_dims
        self.num_views = len(view_slices) if view_slices else 1
        
        # 为JIHL模块预留的组件
        self.jihl_enabled = False  # 是否启用JIHL模块
        self.jihl_module = None
        
        # 初始化JIHL模块
        if view_dims is not None:
            try:
                from jihl_module import JIHLModule
                self.jihl_module = JIHLModule(
                    feature_dim=input_dim,
                    view_dims=view_dims,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                )
            except ImportError:
                print("警告：JIHL模块导入失败，将使用基础模型")
        
    def set_jihl_enabled(self, enabled=True):
        """启用或禁用JIHL模块"""
        self.jihl_enabled = enabled
    
    def set_jihl_module(self, jihl_module):
        """设置JIHL模块"""
        self.jihl_module = jihl_module
        self.jihl_enabled = True
        
    def get_view_features(self, X, view_idx):
        """获取指定视图的特征"""
        if self.view_slices is None or view_idx >= len(self.view_slices):
            return X
        
        start_idx, end_idx = self.view_slices[view_idx]
        return X[:, start_idx:end_idx]
    
    def get_all_view_features(self, X):
        """获取所有视图的特征"""
        if self.view_slices is None:
            return [X]
        
        view_features = []
        for i in range(self.num_views):
            view_feat = self.get_view_features(X, i)
            view_features.append(view_feat)
        return view_features
    
    def calculate_view_confidence(self, h_samples, view_idx):
        """计算指定视图的置信度"""
        # 简化实现：使用特征向量的L2范数作为置信度
        if self.view_slices is None:
            return torch.ones(h_samples.shape[0], device=h_samples.device)
        
        view_features = self.get_view_features(h_samples, view_idx)
        confidence = torch.norm(view_features, dim=1)
        return torch.sigmoid(confidence)
    
    def calculate_view_conflict(self, h_samples, view_idx):
        """计算指定视图的冲突度"""
        # 简化实现：使用与其他视图的平均余弦距离
        if self.view_slices is None or self.num_views <= 1:
            return torch.zeros(h_samples.shape[0], device=h_samples.device)
        
        current_view = self.get_view_features(h_samples, view_idx)
        other_views = []
        
        for i in range(self.num_views):
            if i != view_idx:
                other_view = self.get_view_features(h_samples, i)
                other_views.append(other_view)
        
        if not other_views:
            return torch.zeros(h_samples.shape[0], device=h_samples.device)
        
        # 计算与其他视图的平均余弦距离
        other_mean = torch.stack(other_views).mean(dim=0)
        cosine_sim = F.cosine_similarity(current_view, other_mean, dim=1)
        conflict = (1 - cosine_sim) / 2  # 归一化到[0,1]
        return conflict
        
    def forward_features(self, X, H):
        """
        前向传播到特征层（用于对比损失）
        """
        if self.use_hypergat:
            # 使用HyperGAT进行特征提取
            h = self.hypergat(X, H)
        else:
            # 使用简单的超图卷积（向后兼容）
        H = H.to(X.device)
        d_v = H.sum(1)  # [N]
        d_e = H.sum(0)  # [E]
        d_v = torch.clamp(d_v, min=1.0)
        d_e = torch.clamp(d_e, min=1.0)
        H_norm = H / d_e  # Column normalization
        Xh = H_norm @ H.t()  # [N, N]
        Xh = Xh @ X
        Xh = Xh / d_v.unsqueeze(1)  # Row normalization
        h = torch.relu(self.fc1(Xh))
        return h
    
    def forward(self, X, H, mask=None, H_lowconf=None, H_trans=None):
        """
        重新设计的前向传播架构
        - 传入GCN的特征始终是initial_imputed_features
        - JIHL模块并行运行，用于计算重构损失
        """
        # 始终使用原始特征进行GCN前向传播
        h = self.forward_features(X, H)
        h = self.dropout(h)
        out = self.fc2(h)
        
        # 如果启用JIHL模块，并行计算重构特征
        if self.jihl_enabled and self.jihl_module is not None and mask is not None:
            # 将视图掩码转换为特征掩码
            feature_mask = self.convert_view_mask_to_feature_mask(mask, X.shape[1])
            # 使用当前特征嵌入进行JIHL精炼
            X_refined = self.jihl_module(X, h, feature_mask, H_lowconf, H_trans)
            # 将重构特征存储为属性，供损失计算使用
            self.X_refined = X_refined
        else:
            self.X_refined = None
            
        return out
    
    def convert_view_mask_to_feature_mask(self, view_mask, feature_dim):
        """
        将视图掩码转换为特征掩码
        Args:
            view_mask: [N, num_views] 视图掩码
            feature_dim: 特征维度
        Returns:
            feature_mask: [N, feature_dim] 特征掩码
        """
        batch_size, num_views = view_mask.shape
        feature_mask = torch.zeros(batch_size, feature_dim, device=view_mask.device)
        
        # 如果视图切片未定义，假设每个视图的特征维度相等
        if self.view_slices is None:
            view_dim = feature_dim // num_views
            for i in range(num_views):
                start_idx = i * view_dim
                end_idx = (i + 1) * view_dim if i < num_views - 1 else feature_dim
                feature_mask[:, start_idx:end_idx] = view_mask[:, i].unsqueeze(1)
        else:
            # 使用预定义的视图切片
            for i, (start_idx, end_idx) in enumerate(self.view_slices):
                if i < num_views:
                    feature_mask[:, start_idx:end_idx] = view_mask[:, i].unsqueeze(1)
        
        return feature_mask
    
    def get_refined_features(self):
        """获取JIHL模块重构的特征（用于重构损失计算）"""
        return self.X_refined
    
    def get_features(self, X, H):
        """
        获取特征表示（用于对比损失）
        """
        return self.forward_features(X, H)