import numpy as np
import torch
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import re

def load_data_auto(mat_file):
    """自动加载不同格式的.mat文件"""
    data = sio.loadmat(mat_file)
    
    # 标准格式
    if 'X_train' in data and 'y_train' in data:
        X_train = data['X_train']
        y_train = data['y_train'].flatten() - 1
        X_test = data['X_test']
        y_test = data['y_test'].flatten() - 1
        m_train = data.get('m_train', data.get('mask_train'))
        m_test = data.get('m_test', data.get('mask_test'))
    
    # 多视图格式
    elif 'X' in data and 'gt_train' in data:
        X_views = data['X']
        if isinstance(X_views, np.ndarray) and X_views.dtype == 'O':
            X_views = [X_views[0, i] for i in range(X_views.shape[1])]
        X_train = np.concatenate([v['train'] if isinstance(v, dict) and 'train' in v else v for v in X_views], axis=1)
        X_test = np.concatenate([v['test'] if isinstance(v, dict) and 'test' in v else v for v in X_views], axis=1)
        y_train = data['gt_train'].flatten() - 1
        y_test = data['gt_test'].flatten() - 1
        m_train = data.get('m_train', data.get('mask_train'))
        m_test = data.get('m_test', data.get('mask_test'))
    
    # 分离视图格式
    elif 'gt_train' in data and any(k.startswith('x1_train') for k in data.keys()):
        x_train_list = []
        x_test_list = []
        i = 1
        while f'x{i}_train' in data:
            x_train_list.append(data[f'x{i}_train'])
            x_test_list.append(data[f'x{i}_test'])
            i += 1
        X_train = np.concatenate(x_train_list, axis=1)
        X_test = np.concatenate(x_test_list, axis=1)
        y_train = data['gt_train'].flatten() - 1
        y_test = data['gt_test'].flatten() - 1
        m_train = None
        m_test = None
    
    else:
        raise ValueError(f"Unsupported .mat file format: {mat_file}")
    
    # 确保标签是整数类型
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    view_dims = [216, 76, 64, 6, 47, 240]
    view_slices = []
    start = 0
    for d in view_dims:
        view_slices.append((start, start + d))
        start += d
    return X_train, y_train, X_test, y_test, m_train, m_test, view_slices

def preprocess_data(X_train, y_train, X_test, y_test, m_train=None, m_test=None):
    """数据预处理"""
    # 处理NaN值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 标准化YUHJHHHHHHHHHHHH
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 转换为tensor，确保标签是长整型
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    if m_train is not None:
        m_train = torch.tensor(m_train, dtype=torch.float32)
    if m_test is not None:
        m_test = torch.tensor(m_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, m_train, m_test