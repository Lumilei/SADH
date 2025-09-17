import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
from sklearn.metrics import accuracy_score

class ExperimentTrainer:
    def __init__(self, config_path=None, config=None):
        """
        初始化实验训练器
        Args:
            config_path: 配置文件路径
            config: 配置字典对象
        """
        if config is not None:
            self.config = config
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 训练参数
        self.lr = float(self.config['training']['lr'])
        self.epochs = int(self.config['training']['epochs'])
        self.weight_decay = float(self.config['training']['weight_decay'])
        
        # 损失函数参数
        self.use_cls = self.config['loss']['use_cls']
        self.use_contrastive = self.config['loss'].get('use_contrastive', False)
        self.lambda_con = float(self.config['loss'].get('lambda_con', 0.1))
        self.tau = float(self.config['loss'].get('tau', 0.07))
        
        print(f"训练配置: lr={self.lr}, epochs={self.epochs}, weight_decay={self.weight_decay}")
        print(f"损失函数: use_cls={self.use_cls}, use_contrastive={self.use_contrastive}")
        if self.use_contrastive:
            print(f"对比损失: lambda_con={self.lambda_con}, tau={self.tau}")
    
    def contrastive_loss(self, features, labels):
        """
        计算对比损失
        Args:
            features: 特征向量 [N, D]
            labels: 标签 [N]
        Returns:
            contrastive_loss: 对比损失
        """
        features = nn.functional.normalize(features, dim=1)
        labels = labels.to(features.device)
        similarity_matrix = torch.matmul(features, features.T) / self.tau
        labels = labels.unsqueeze(0)
        label_matrix = (labels == labels.T).float()
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob = (log_prob * label_matrix).sum(dim=1) / (label_matrix.sum(dim=1) + 1e-8)
        contrastive_loss = -mean_log_prob.mean()
        return contrastive_loss
    
    def train_model(self, model, X_train, y_train, X_test, y_test, H_train, H_test, 
                   mask_train=None, mask_test=None, H_lowconf_train=None, H_trans_train=None):
        """
        训练模型
        Args:
            model: 模型
            X_train, y_train: 训练数据和标签
            X_test, y_test: 测试数据和标签
            H_train, H_test: 训练和测试超图
            mask_train, mask_test: 训练和测试的缺失掩码
            H_lowconf_train: 训练集的低置信度超边
            H_trans_train: 训练集的信息传递超边
        """
        model = model.to(self.device)
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        H_train = H_train.to(self.device)
        H_test = H_test.to(self.device)
        
        # 处理缺失掩码
        if mask_train is not None:
            mask_train = mask_train.to(self.device)
        if mask_test is not None:
            mask_test = mask_test.to(self.device)
        
        # 处理超边矩阵
        if H_lowconf_train is not None:
            H_lowconf_train = H_lowconf_train.to(self.device)
        if H_trans_train is not None:
            H_trans_train = H_trans_train.to(self.device)
        
        # 启用JIHL模块（如果模型支持）
        if hasattr(model, 'set_jihl_enabled'):
            model.set_jihl_enabled(True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        print(f"开始训练，总轮数: {self.epochs}")
        print("="*60)
        
        best_accuracy = 0.0
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            
            try:
                # 前向传播（新架构：始终使用initial_imputed_features）
                if mask_train is not None:
                    outputs = model(X_train, H_train, mask_train, H_lowconf_train, H_trans_train)
                else:
                    outputs = model(X_train, H_train)
                
                # 分类损失
                cls_loss = criterion(outputs, y_train)
                
                # 对比损失
                contrastive_loss = 0.0
                if self.use_contrastive:
                    features = model.get_features(X_train, H_train)
                    contrastive_loss = self.contrastive_loss(features, y_train)
                
                # 重构损失（JIHL模块）
                recon_loss = 0.0
                if mask_train is not None and hasattr(model, 'get_refined_features'):
                    X_refined = model.get_refined_features()
                    if X_refined is not None:
                        # 计算重构损失：JIHL输出与真实特征的MSE
                        # 只对缺失的特征计算损失
                        missing_mask = ~mask_train.bool()
                        if missing_mask.any():
                            recon_loss = F.mse_loss(
                                X_refined[missing_mask], 
                                X_train[missing_mask]
                            )
                
                # 总损失：分类损失 + 对比损失 + 重构损失
                lambda_recon = 1e-4  # 设置很小的重构损失权重
                total_loss = cls_loss + self.lambda_con * contrastive_loss + lambda_recon * recon_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                # 评估
                model.eval()
                with torch.no_grad():
                    if mask_test is not None:
                        test_outputs = model(X_test, H_test, mask_test)
                    else:
                        test_outputs = model(X_test, H_test)
                    
                    _, predicted = torch.max(test_outputs.data, 1)
                    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                
                if (epoch + 1) % 10 == 0:
                    loss_info = f"Epoch [{epoch+1}/{self.epochs}], Cls: {cls_loss.item():.4f}"
                    if self.use_contrastive:
                        loss_info += f", Con: {contrastive_loss.item():.4f}"
                    if recon_loss > 0:
                        loss_info += f", Recon: {recon_loss.item():.6f}"
                    loss_info += f", Total: {total_loss.item():.4f}, Acc: {accuracy:.4f}"
                    print(loss_info)
                    
            except Exception as e:
                print(f"训练失败: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return 0.0
        
        print(f"训练完成，最佳准确率: {best_accuracy:.4f}")
        return best_accuracy
    
    def evaluate_model(self, model, X_test, y_test, H_test):
        """
        评估模型
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            H_test: 测试超图
        Returns:
            accuracy: 准确率
            predictions: 预测结果
        """
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            H_test = H_test.to(self.device)
            
            outputs = model(X_test, H_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
            
            return accuracy, predicted.cpu().numpy()
    
    def get_model_predictions(self, model, X, H):
        """
        获取模型预测结果
        Args:
            model: 模型
            X: 输入特征
            H: 超图
        Returns:
            predictions: 预测结果
            probabilities: 预测概率
        """
        model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            H = H.to(self.device)
            
            outputs = model(X, H)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            return predictions.cpu().numpy(), probabilities.cpu().numpy()