#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超图注意力卷积层 (HyperGAT)
实现带有注意力机制的超图卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HyperGATConv(nn.Module):
    """超图注意力卷积层"""
    
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1, alpha=0.2):
        super(HyperGATConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # 确保输出维度能被头数整除
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads
        
        # 线性变换层
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # 注意力参数
        self.attention = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        
        # 输出偏置
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.zeros_(self.bias)
    
    def forward(self, X, H):
        """
        前向传播
        Args:
            X: 节点特征 [N, in_dim]
            H: 超图关联矩阵 [N, E]
        Returns:
            out: 更新后的节点特征 [N, out_dim]
        """
        N, E = H.shape
        
        # 线性变换
        X_transformed = self.W(X)  # [N, out_dim]
        
        # 重塑为多头形式
        X_heads = X_transformed.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # 计算超边内的注意力
        attention_scores = self._compute_attention_scores(X_heads, H)
        
        # 应用注意力权重
        out_heads = self._apply_attention(X_heads, H, attention_scores)
        
        # 合并多头
        out = out_heads.view(N, self.out_dim)
        
        # 添加偏置
        out = out + self.bias
        
        return out
    
    def _compute_attention_scores(self, X_heads, H):
        """
        计算注意力分数
        Args:
            X_heads: 多头节点特征 [N, num_heads, head_dim]
            H: 超图关联矩阵 [N, E]
        Returns:
            attention_scores: 注意力分数 [E, num_heads, max_degree]
        """
        N, E = H.shape
        device = X_heads.device
        
        # 计算每个超边的最大度数
        degrees = H.sum(dim=0)  # [E]
        max_degree = int(degrees.max().item())
        
        # 初始化注意力分数
        attention_scores = torch.zeros(E, self.num_heads, max_degree, device=device)
        
        for e in range(E):
            # 获取超边e中的节点
            nodes_in_edge = torch.where(H[:, e] > 0)[0]  # 节点索引
            degree = len(nodes_in_edge)
            
            if degree == 0:
                continue
            
            # 获取超边内节点的特征
            edge_features = X_heads[nodes_in_edge]  # [degree, num_heads, head_dim]
            
            # 计算注意力分数
            for i in range(degree):
                # 当前节点与超边内所有节点的注意力
                current_node = edge_features[i:i+1]  # [1, num_heads, head_dim]
                all_nodes = edge_features  # [degree, num_heads, head_dim]
                
                # 拼接特征
                concat_features = torch.cat([
                    current_node.expand(degree, -1, -1),  # [degree, num_heads, head_dim]
                    all_nodes  # [degree, num_heads, head_dim]
                ], dim=-1)  # [degree, num_heads, 2*head_dim]
                
                # 计算注意力分数
                scores = torch.sum(
                    self.attention * concat_features, 
                    dim=-1
                )  # [degree, num_heads]
                
                # 应用LeakyReLU
                scores = F.leaky_relu(scores, negative_slope=self.alpha)
                
                # 存储注意力分数
                attention_scores[e, :, i] = scores[i]
        
        return attention_scores
    
    def _apply_attention(self, X_heads, H, attention_scores):
        """
        应用注意力权重
        Args:
            X_heads: 多头节点特征 [N, num_heads, head_dim]
            H: 超图关联矩阵 [N, E]
            attention_scores: 注意力分数 [E, num_heads, max_degree]
        Returns:
            out_heads: 更新后的多头特征 [N, num_heads, head_dim]
        """
        N, E = H.shape
        device = X_heads.device
        
        # 初始化输出
        out_heads = torch.zeros_like(X_heads)
        
        # 计算每个超边的最大度数
        degrees = H.sum(dim=0)  # [E]
        max_degree = int(degrees.max().item())
        
        for e in range(E):
            # 获取超边e中的节点
            nodes_in_edge = torch.where(H[:, e] > 0)[0]  # 节点索引
            degree = len(nodes_in_edge)
            
            if degree == 0:
                continue
            
            # 获取超边内节点的特征
            edge_features = X_heads[nodes_in_edge]  # [degree, num_heads, head_dim]
            
            # 获取注意力权重
            edge_attention = attention_scores[e, :, :degree]  # [num_heads, degree]
            
            # 应用softmax归一化
            edge_attention = F.softmax(edge_attention, dim=-1)  # [num_heads, degree]
            
            # 应用dropout
            edge_attention = F.dropout(edge_attention, p=self.dropout, training=self.training)
            
            # 加权聚合
            # edge_attention: [num_heads, degree]
            # edge_features: [degree, num_heads, head_dim]
            # 需要调整维度以进行批量矩阵乘法
            edge_attention = edge_attention.unsqueeze(-1)  # [num_heads, degree, 1]
            edge_features = edge_features.transpose(0, 1)  # [num_heads, degree, head_dim]
            
            aggregated = torch.sum(edge_attention * edge_features, dim=1)  # [num_heads, head_dim]
            
            # 将聚合结果分配给超边内的所有节点
            for i, node_idx in enumerate(nodes_in_edge):
                out_heads[node_idx] += aggregated
        
        return out_heads


class MultiLayerHyperGAT(nn.Module):
    """多层超图注意力网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 num_heads=8, dropout=0.1, alpha=0.2):
        super(MultiLayerHyperGAT, self).__init__()
        self.num_layers = num_layers
        
        # 构建多层
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(HyperGATConv(
            input_dim, hidden_dim, num_heads, dropout, alpha
        ))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(HyperGATConv(
                hidden_dim, hidden_dim, num_heads, dropout, alpha
            ))
        
        # 最后一层
        if num_layers > 1:
            self.layers.append(HyperGATConv(
                hidden_dim, output_dim, num_heads, dropout, alpha
            ))
        
        # 激活函数
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, H):
        """
        前向传播
        Args:
            X: 输入特征 [N, input_dim]
            H: 超图关联矩阵 [N, E]
        Returns:
            out: 输出特征 [N, output_dim]
        """
        x = X
        
        for i, layer in enumerate(self.layers):
            x = layer(x, H)
            
            # 除了最后一层，都应用激活函数和dropout
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x


def test_hypergat():
    """测试HyperGAT层"""
    print("🧪 测试HyperGAT层...")
    
    # 创建测试数据
    N, E, in_dim, out_dim = 10, 5, 64, 128
    X = torch.randn(N, in_dim)
    H = torch.randint(0, 2, (N, E)).float()
    
    print(f"输入特征形状: {X.shape}")
    print(f"超图关联矩阵形状: {H.shape}")
    
    # 测试单层HyperGAT
    hypergat = HyperGATConv(in_dim, out_dim, num_heads=8)
    out = hypergat(X, H)
    print(f"单层输出形状: {out.shape}")
    
    # 测试多层HyperGAT
    ml_hypergat = MultiLayerHyperGAT(in_dim, 64, out_dim, num_layers=2)
    out_ml = ml_hypergat(X, H)
    print(f"多层输出形状: {out_ml.shape}")
    
    print("✅ HyperGAT测试成功!")


if __name__ == "__main__":
    test_hypergat() 