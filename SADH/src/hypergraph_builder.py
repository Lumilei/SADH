import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_instview_h_old(n_samples, n_views, m=None):
    """
    旧的实例-视图超边构建函数（重命名以避免冲突）
    """
    # 这个函数暂时不使用，重命名以避免冲突
    pass

def build_category_hyperedge(X_train, y_train, X_test, y_test):
    """
    构建类别超边
    """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    # 确保标签是 Tensor 类型
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 获取所有唯一类别
    all_labels = torch.cat([y_train, y_test])
    unique_labels = torch.unique(all_labels)
    num_classes = len(unique_labels)
    
    print(f"类别超边 - 类别数: {num_classes}, 训练集: {N_train}, 测试集: {N_test}")
    
    # 构建训练集类别超边
    H_train = torch.zeros(N_train, num_classes)
    for i, label in enumerate(unique_labels):
        mask = (y_train == label)
        H_train[mask, i] = 1.0
    
    # 构建测试集类别超边
    H_test = torch.zeros(N_test, num_classes)
    for i, label in enumerate(unique_labels):
        mask = (y_test == label)
        H_test[mask, i] = 1.0
    
    print(f"类别超边构建完成 - H_train: {H_train.shape}, H_test: {H_test.shape}")
    print(f"训练集超边密度: {H_train.sum() / H_train.numel():.4f}")
    print(f"测试集超边密度: {H_test.sum() / H_test.numel():.4f}")
    
    return H_train, H_test

def build_lowconf_hyperedge(X_train, y_train, X_test, y_test, config):
    """
    构建低置信度超边
    """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    # 确保数据是 Tensor 类型
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    
    # 获取B参数
    B = config['hypergraph'].get('lowconf_B', 20)
    
    print(f"构建低置信度超边 - B: {B}")
    
    # 使用简单的置信度估计（基于特征方差）
    # 确保数据没有 NaN 或 Inf
    X_train_clean = torch.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_clean = torch.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    train_conf = torch.std(X_train_clean, dim=1)
    test_conf = torch.std(X_test_clean, dim=1)
    
    # 选择置信度最低的B个样本
    _, train_indices = torch.topk(train_conf, min(B, N_train), largest=False)
    _, test_indices = torch.topk(test_conf, min(B, N_test), largest=False)
    
    # 构建超边
    H_train = torch.zeros(N_train, 1)
    H_test = torch.zeros(N_test, 1)
    
    H_train[train_indices, 0] = 1.0
    H_test[test_indices, 0] = 1.0
    
    print(f"低置信度超边构建完成 - H_train: {H_train.shape}, H_test: {H_test.shape}")
    print(f"训练集超边密度: {H_train.sum() / H_train.numel():.4f}")
    print(f"测试集超边密度: {H_test.sum() / H_test.numel():.4f}")
    
    return H_train, H_test

def build_infotrans_hyperedge(X_train, y_train, X_test, y_test, config):
    """
    构建信息传递超边
    """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    # 确保数据是 Tensor 类型
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    
    # 获取k_nn参数
    k_nn = config['hypergraph'].get('k_nn', 5)
    
    print(f"构建信息传递超边 - k_nn: {k_nn}")
    
    # 数据预处理：处理 NaN 和 Inf
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    
    # 替换 NaN 和 Inf
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    
    # 计算训练集内部的k-NN
    from sklearn.neighbors import NearestNeighbors
    
    try:
        # 训练集内部k-NN
        nbrs_train = NearestNeighbors(n_neighbors=min(k_nn+1, N_train), algorithm='auto').fit(X_train_scaled)
        distances_train, indices_train = nbrs_train.kneighbors(X_train_scaled)
        
        # 测试集内部k-NN
        nbrs_test = NearestNeighbors(n_neighbors=min(k_nn+1, N_test), algorithm='auto').fit(X_test_scaled)
        distances_test, indices_test = nbrs_test.kneighbors(X_test_scaled)
        
        # 构建超边（每个样本作为一个超边，包含其k个最近邻）
        H_train = torch.zeros(N_train, N_train)
        H_test = torch.zeros(N_test, N_test)
        
        for i in range(N_train):
            neighbors = indices_train[i][1:]  # 排除自己
            H_train[i, neighbors] = 1.0
        
        for i in range(N_test):
            neighbors = indices_test[i][1:]  # 排除自己
            H_test[i, neighbors] = 1.0
        
        print(f"信息传递超边构建完成 - H_train: {H_train.shape}, H_test: {H_test.shape}")
        print(f"训练集超边密度: {H_train.sum() / H_train.numel():.4f}")
        print(f"测试集超边密度: {H_test.sum() / H_test.numel():.4f}")
        
        return H_train, H_test
        
    except Exception as e:
        print(f"信息传递超边构建失败: {e}")
        print("使用简化的信息传递超边")
        
        # 使用简化的方法：基于欧氏距离的最近邻
        H_train = torch.zeros(N_train, N_train)
        H_test = torch.zeros(N_test, N_test)
        
        # 计算训练集内部的相似度
        for i in range(N_train):
            distances = torch.norm(X_train - X_train[i], dim=1)
            _, indices = torch.topk(distances, min(k_nn+1, N_train), largest=False)
            neighbors = indices[1:]  # 排除自己
            H_train[i, neighbors] = 1.0
        
        # 计算测试集内部的相似度
        for i in range(N_test):
            distances = torch.norm(X_test - X_test[i], dim=1)
            _, indices = torch.topk(distances, min(k_nn+1, N_test), largest=False)
            neighbors = indices[1:]  # 排除自己
            H_test[i, neighbors] = 1.0
        
        print(f"简化信息传递超边构建完成 - H_train: {H_train.shape}, H_test: {H_test.shape}")
        print(f"训练集超边密度: {H_train.sum() / H_train.numel():.4f}")
        print(f"测试集超边密度: {H_test.sum() / H_test.numel():.4f}")
        
        return H_train, H_test

def build_instview_h(X_train, X_test, config):
    """
    构建实例-视图超边
    """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    print("构建实例-视图超边（暂未实现）")
    H_train = torch.zeros(N_train, 0)
    H_test = torch.zeros(N_test, 0)
    
    return H_train, H_test

def build_hypergraph(X_train, y_train, X_test, y_test, config):
    """
    构建超图
    Returns:
        H_train, H_test: 合并后的超图
        H_lowconf_train, H_lowconf_test: 低置信度超边（用于JIHL）
        H_trans_train, H_trans_test: 信息传递超边（用于JIHL）
    """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    print(f"构建超图 - N_train: {N_train}, N_test: {N_test}")
    
    # 检查配置
    if 'hypergraph' not in config:
        print("❌ 错误：配置中缺少 'hypergraph' 部分")
        print("当前配置的键：", list(config.keys()))
        raise KeyError("配置中缺少 'hypergraph' 部分")
    
    hypergraph_config = config['hypergraph']
    print(f"超图配置：{hypergraph_config}")
    
    H_train_list = []
    H_test_list = []
    
    # 初始化JIHL相关的超边
    H_lowconf_train = None
    H_lowconf_test = None
    H_trans_train = None
    H_trans_test = None
    
    # 类别超边
    if hypergraph_config.get('use_category', False):
        print("构建类别超边...")
        H_train_cat, H_test_cat = build_category_hyperedge(X_train, y_train, X_test, y_test)
        print(f"类别超边 - H_train_cat: {H_train_cat.shape}, H_test_cat: {H_test_cat.shape}")
        
        # 检查 shape
        assert H_train_cat.shape[0] == N_train, f"类别超边训练集shape错误: {H_train_cat.shape[0]} != {N_train}"
        assert H_test_cat.shape[0] == N_test, f"类别超边测试集shape错误: {H_test_cat.shape[0]} != {N_test}"
        
        H_train_list.append(H_train_cat)
        H_test_list.append(H_test_cat)
    
    # 低置信度超边
    if hypergraph_config.get('use_lowconf', False):
        print("构建低置信度超边...")
        H_train_lowconf, H_test_lowconf = build_lowconf_hyperedge(X_train, y_train, X_test, y_test, config)
        print(f"低置信度超边 - H_train_lowconf: {H_train_lowconf.shape}, H_test_lowconf: {H_test_lowconf.shape}")
        
        # 检查 shape
        assert H_train_lowconf.shape[0] == N_train, f"低置信度超边训练集shape错误: {H_train_lowconf.shape[0]} != {N_train}"
        assert H_test_lowconf.shape[0] == N_test, f"低置信度超边测试集shape错误: {H_test_lowconf.shape[0]} != {N_test}"
        
        H_train_list.append(H_train_lowconf)
        H_test_list.append(H_test_lowconf)
        
        # 保存低置信度超边用于JIHL
        H_lowconf_train = H_train_lowconf
        H_lowconf_test = H_test_lowconf
    
    # 信息传递超边
    if hypergraph_config.get('use_infotrans', False):
        print("构建信息传递超边...")
        H_train_infotrans, H_test_infotrans = build_infotrans_hyperedge(X_train, y_train, X_test, y_test, config)
        print(f"信息传递超边 - H_train_infotrans: {H_train_infotrans.shape}, H_test_infotrans: {H_test_infotrans.shape}")
        
        # 检查 shape
        assert H_train_infotrans.shape[0] == N_train, f"信息传递超边训练集shape错误: {H_train_infotrans.shape[0]} != {N_train}"
        assert H_test_infotrans.shape[0] == N_test, f"信息传递超边测试集shape错误: {H_test_infotrans.shape[0]} != {N_test}"
        
        H_train_list.append(H_train_infotrans)
        H_test_list.append(H_test_infotrans)
        
        # 保存信息传递超边用于JIHL
        H_trans_train = H_train_infotrans
        H_trans_test = H_test_infotrans
    
    # 实例-视图超边
    if hypergraph_config.get('use_instview', False):
        print("构建实例-视图超边...")
        H_train_instview, H_test_instview = build_instview_h(X_train, X_test, config)
        print(f"实例-视图超边 - H_train_instview: {H_train_instview.shape}, H_test_instview: {H_test_instview.shape}")
        
        # 检查 shape
        assert H_train_instview.shape[0] == N_train, f"实例-视图超边训练集shape错误: {H_train_instview.shape[0]} != {N_train}"
        assert H_test_instview.shape[0] == N_test, f"实例-视图超边测试集shape错误: {H_test_instview.shape[0]} != {N_test}"
        
        H_train_list.append(H_train_instview)
        H_test_list.append(H_test_instview)
    
    # 合并所有超边
    print(f"准备合并超边 - H_train_list长度: {len(H_train_list)}, H_test_list长度: {len(H_test_list)}")
    
    if H_train_list:
        print("合并训练集超边...")
        for i, h in enumerate(H_train_list):
            print(f"  H_train_list[{i}] shape: {h.shape}")
        H_train = torch.cat(H_train_list, dim=1)
        print(f"合并后 H_train shape: {H_train.shape}")
        
        print("合并测试集超边...")
        for i, h in enumerate(H_test_list):
            print(f"  H_test_list[{i}] shape: {h.shape}")
        H_test = torch.cat(H_test_list, dim=1)
        print(f"合并后 H_test shape: {H_test.shape}")
    else:
        # 如果没有超边，创建空的超图
        print("没有超边，创建空超图")
        H_train = torch.zeros(N_train, 0)
        H_test = torch.zeros(N_test, 0)
    
    print(f"最终超图 - H_train: {H_train.shape}, H_test: {H_test.shape}")
    
    # 返回合并后的超图和JIHL相关的超边
    return H_train, H_test, H_lowconf_train, H_lowconf_test, H_trans_train, H_trans_test