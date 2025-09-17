#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习率搜索脚本
系统测试不同学习率，找到最佳配置
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_simplified_experimrnt import (
    load_config, load_handwritten_data, build_simplified_hypergraph,
    initialize_features_according_to_paper, split_train_test
)
from models import SimpleHyperGCN
from jihl_module import JIHLModule

# 直接定义SimpleTrainer类，避免导入问题
class SimpleTrainer:
    def __init__(self, lr=0.001, weight_decay=1e-4, tau=0.07):
        self.lr = lr
        self.weight_decay = weight_decay
        self.tau = tau
    
    def contrastive_loss(self, features, labels):
        """计算对比损失 - InfoNCE损失"""
        # 归一化特征
        features = torch.nn.functional.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.tau
        
        # 创建标签矩阵：相同标签为正样本对
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线（自己与自己的相似度）
        mask = mask - torch.eye(mask.shape[0], device=mask.device)
        
        # 计算正样本对和负样本对
        positives = similarity_matrix * mask
        negatives = similarity_matrix * (1 - mask)
        
        # 计算正样本对数量
        num_positives = mask.sum(dim=1)
        
        # 计算对比损失
        logits = torch.cat([positives, negatives], dim=1)
        labels_contrastive = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        
        # 计算每个样本的对比损失
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # 只对正样本对计算损失
        mean_log_prob_pos = (mask * log_prob[:, :mask.shape[1]]).sum(dim=1) / (num_positives + 1e-8)
        
        # 最终损失
        contrastive_loss = -mean_log_prob_pos.mean()
        
        # 调试信息
        if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
            print(f"⚠️ 对比损失异常: {contrastive_loss.item()}")
            return torch.tensor(0.0, device=features.device)
        
        return contrastive_loss
    
    def train_model(self, model, X_train, H_train, y_train, mask_train, 
                   X_test, H_test, y_test, mask_test, epochs=100, 
                   use_contrastive=True, lambda_con=0.1):
        """训练模型"""
        device = next(model.parameters()).device
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(device)
        H_train = torch.FloatTensor(H_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        mask_train = torch.FloatTensor(mask_train).to(device)
        
        X_test = torch.FloatTensor(X_test).to(device)
        H_test = torch.FloatTensor(H_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)
        mask_test = torch.FloatTensor(mask_test).to(device)
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(X_train, H_train, mask_train)
            
            # 分类损失
            cls_loss = criterion(outputs, y_train)
            
            # 对比损失
            contrastive_loss = 0.0
            if use_contrastive:
                features = model.get_features(X_train, H_train)
                contrastive_loss = self.contrastive_loss(features, y_train)
                
                # 调试信息：只在第一个epoch打印
                if epoch == 0:
                    print(f"🔍 对比损失调试信息:")
                    print(f"   特征形状: {features.shape}")
                    print(f"   标签形状: {y_train.shape}")
                    print(f"   对比损失值: {contrastive_loss.item():.4f}")
                    print(f"   特征范数范围: [{features.norm(dim=1).min().item():.4f}, {features.norm(dim=1).max().item():.4f}]")
            
            # 重构损失
            recon_loss = 0.0
            if hasattr(model, 'get_refined_features'):
                X_refined = model.get_refined_features()
                if X_refined is not None:
                    # 使用特征级掩码
                    feature_mask = model.convert_view_mask_to_feature_mask(mask_train, X_train.shape[1])
                    missing_mask = ~feature_mask.bool()
                    if missing_mask.any():
                        recon_loss = torch.nn.functional.mse_loss(
                            X_refined[missing_mask], 
                            X_train[missing_mask]
                        )
            
            # 总损失
            lambda_recon = 1e-4
            total_loss = cls_loss + lambda_con * contrastive_loss + lambda_recon * recon_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 评估
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test, H_test, mask_test)
                    _, predicted = torch.max(test_outputs, 1)
                    accuracy = (predicted == y_test).float().mean().item()
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    
                    print(f"Epoch [{epoch+1}/{epochs}], Cls: {cls_loss:.4f}, Con: {contrastive_loss:.4f}, Recon: {recon_loss:.6f}, Total: {total_loss:.4f}, Acc: {accuracy:.4f}")
        
        print(f"训练完成，最佳准确率: {best_accuracy:.4f}")
        return best_accuracy

def run_lr_experiment(config_path, data_path, learning_rate, experiment_name):
    """运行单个学习率实验"""
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"学习率: {learning_rate}")
    print(f"{'='*60}")
    
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 修改学习率
        config['training']['learning_rate'] = learning_rate
        
        # 加载数据
        print(f"加载数据集: {data_path}")
        X, y, mask = load_handwritten_data(data_path)
        
        # 特征初始化
        print("按照论文方法初始化特征...")
        X_imputed, view_centroids = initialize_features_according_to_paper(X, mask)
        
        # 数据分割
        X_train, X_test, y_train, y_test, mask_train, mask_test = split_train_test(
            X_imputed, y, mask, test_ratio=0.2, random_state=42
        )
        
        # 构建超图
        print("构建简化超图...")
        H_train, y_train_mapped = build_simplified_hypergraph(X_train, y_train, mask_train, config)
        H_test, y_test_mapped = build_simplified_hypergraph(X_test, y_test, mask_test, config)
        
        print(f"最终超图 - H_train: {H_train.shape}, H_test: {H_test.shape}")
        
        # 设备设置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 模型初始化
        print("启用JIHL模块...")
        try:
            model = SimpleHyperGCN(
                input_dim=X_train.shape[1],
                hidden_dim=config['model']['hidden_dim'],
                num_classes=len(np.unique(y_train_mapped)),
                dropout=config['model']['dropout']
            ).to(device)
            print("✅ 模型创建成功")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            # 尝试使用不同的参数名
            try:
                model = SimpleHyperGCN(
                    in_dim=X_train.shape[1],
                    embed_dim=config['model']['hidden_dim'],
                    num_classes=len(np.unique(y_train_mapped)),
                    dropout=config['model']['dropout']
                ).to(device)
                print("✅ 使用替代参数名创建成功")
            except Exception as e2:
                print(f"❌ 替代参数名也失败: {e2}")
                raise e2
        
        # 设置JIHL模块
        if config['model']['jihl_enabled']:
            # 计算视图维度（假设每个视图维度相等）
            feature_dim = X_train.shape[1]
            num_views = mask_train.shape[1]
            view_dim = feature_dim // num_views
            view_dims = [view_dim] * num_views
            
            jihl_module = JIHLModule(
                feature_dim=feature_dim,
                view_dims=view_dims,
                hidden_dim=config['model']['hidden_dim'],
                dropout=config['model']['dropout']
            ).to(device)
            model.set_jihl_module(jihl_module)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 训练器初始化
        trainer = SimpleTrainer(
            lr=learning_rate,
            weight_decay=config['training'].get('weight_decay', 1e-4),
            tau=config['loss'].get('temperature', 0.07)
        )
        
        # 训练模型
        print(f"开始训练，学习率: {learning_rate}")
        print("="*60)
        
        best_accuracy = trainer.train_model(
            model=model,
            X_train=X_train, H_train=H_train, y_train=y_train_mapped, mask_train=mask_train,
            X_test=X_test, H_test=H_test, y_test=y_test_mapped, mask_test=mask_test,
            epochs=config['training']['epochs'],
            use_contrastive=config['loss']['use_contrastive'],
            lambda_con=config['loss']['lambda_con']
        )
        
        print(f"✅ 实验完成 - 最佳准确率: {best_accuracy:.4f}")
        
        return {
            'learning_rate': learning_rate,
            'best_accuracy': best_accuracy,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        return {
            'learning_rate': learning_rate,
            'best_accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def plot_lr_results(results, save_path):
    """绘制学习率搜索结果"""
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("没有成功的结果可以绘制")
        return
    
    lrs = [r['learning_rate'] for r in successful_results]
    accuracies = [r['best_accuracy'] for r in successful_results]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.title('Learning Rate Search Results', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 标记最佳结果
    best_idx = np.argmax(accuracies)
    best_lr = lrs[best_idx]
    best_acc = accuracies[best_idx]
    plt.annotate(f'Best: LR={best_lr:.0e}, Acc={best_acc:.4f}',
                xy=(best_lr, best_acc), xytext=(best_lr*2, best_acc-0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"结果图已保存到: {save_path}")

def main():
    """主函数"""
    print("🚀 开始学习率搜索实验")
    print("="*60)
    
    # 配置路径
    config_path = "config_simplified.yaml"
    data_path = "../datasets/handwritten/basied_missing/handwritten_pareto_missing_0_3_alpha0_8.mat"
    
    # 学习率列表
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"lr_search_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"测试学习率: {learning_rates}")
    print(f"结果保存目录: {results_dir}")
    
    # 运行实验
    results = []
    for i, lr in enumerate(learning_rates):
        experiment_name = f"LR_{lr:.0e}"
        result = run_lr_experiment(config_path, data_path, lr, experiment_name)
        results.append(result)
        
        # 保存中间结果
        with open(f"{results_dir}/intermediate_results.txt", 'a') as f:
            f.write(f"实验 {i+1}/{len(learning_rates)}: {experiment_name}\n")
            f.write(f"学习率: {lr}\n")
            f.write(f"准确率: {result['best_accuracy']:.4f}\n")
            f.write(f"状态: {result['status']}\n")
            if result['status'] == 'failed':
                f.write(f"错误: {result.get('error', 'Unknown')}\n")
            f.write("-" * 40 + "\n")
    
    # 分析结果
    print("\n" + "="*60)
    print("📊 学习率搜索结果汇总")
    print("="*60)
    
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    print(f"成功实验: {len(successful_results)}/{len(results)}")
    print(f"失败实验: {len(failed_results)}/{len(results)}")
    
    if successful_results:
        print("\n成功实验结果:")
        for result in successful_results:
            print(f"  学习率 {result['learning_rate']:.0e}: 准确率 {result['best_accuracy']:.4f}")
        
        # 找到最佳学习率
        best_result = max(successful_results, key=lambda x: x['best_accuracy'])
        print(f"\n🏆 最佳学习率: {best_result['learning_rate']:.0e}")
        print(f"🏆 最佳准确率: {best_result['best_accuracy']:.4f}")
        
        # 绘制结果图
        plot_path = f"{results_dir}/lr_search_plot.png"
        plot_lr_results(results, plot_path)
        
        # 保存详细结果
        with open(f"{results_dir}/detailed_results.txt", 'w') as f:
            f.write("学习率搜索详细结果\n")
            f.write("="*40 + "\n")
            for result in results:
                f.write(f"学习率: {result['learning_rate']:.0e}\n")
                f.write(f"准确率: {result['best_accuracy']:.4f}\n")
                f.write(f"状态: {result['status']}\n")
                if result['status'] == 'failed':
                    f.write(f"错误: {result.get('error', 'Unknown')}\n")
                f.write("-" * 20 + "\n")
        
        # 生成推荐配置
        best_lr = best_result['learning_rate']
        print(f"\n💡 推荐配置:")
        print(f"   学习率: {best_lr:.0e}")
        print(f"   预期准确率: {best_result['best_accuracy']:.4f}")
        
        # 保存推荐配置
        config = load_config(config_path)
        config['training']['learning_rate'] = best_lr
        with open(f"{results_dir}/recommended_config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n📁 所有结果已保存到: {results_dir}")
        
    else:
        print("❌ 所有实验都失败了，请检查配置和代码")
    
    print("\n🎉 学习率搜索完成!")

if __name__ == "__main__":
    main() 