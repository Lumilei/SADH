import os
import glob
import yaml
from data_loader import load_data_auto, preprocess_data
from hypergraph_builder import build_hypergraph
from models import SimpleESADH, SimpleHyperGCN
from trainer import ExperimentTrainer

def main():
    # 加载配置
    trainer = ExperimentTrainer('config.yaml')
    
    # 发现数据集
    data_dir = trainer.config['data']['data_dir']
    mat_files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))
    
    print(f"发现 {len(mat_files)} 个数据集：")
    for f in mat_files:
        print(f"  - {f}")
    
    # 运行实验
    results = {}
    for mat_file in mat_files:
        try:
            print(f"\n处理数据集: {os.path.basename(mat_file)}")
            
            # 加载和预处理数据
            X_train, y_train, X_test, y_test, m_train, m_test = load_data_auto(mat_file)
            X_train, y_train, X_test, y_test, m_train, m_test = preprocess_data(
                X_train, y_train, X_test, y_test, m_train, m_test
            )
            
            # 构建超图
            H_train, H_test = build_hypergraph(
                X_train, y_train, X_test, y_test, m_train, m_test, trainer.config['hypergraph']
            )
            
            # 创建模型
            in_dim = X_train.shape[1]
            num_classes = len(torch.unique(y_train))
            
            if trainer.config['loss'].get('use_contrastive', False):
                model = SimpleESADH(
                    in_dim, 
                    trainer.config['model']['embed_dim'], 
                    num_classes,
                    trainer.config['model']['dropout']
                )
            else:
                model = SimpleHyperGCN(
                    in_dim,
                    trainer.config['model']['hidden_dim'],
                    num_classes,
                    trainer.config['model']['dropout']
                )
            
            # 训练模型
            best_acc, log_lines = trainer.train_model(model, X_train, y_train, X_test, y_test, H_train, H_test)
            results[os.path.basename(mat_file)] = best_acc
            
            # 保存日志
            if trainer.config['experiment']['log_file']:
                with open(trainer.config['experiment']['log_file'], 'a') as f:
                    f.write(f"\n=== {os.path.basename(mat_file)} ===\n")
                    for line in log_lines:
                        f.write(line + '\n')
            
        except Exception as e:
            print(f"❌ 错误：{os.path.basename(mat_file)} - {e}")
            results[os.path.basename(mat_file)] = 0.0
    
    # 输出结果
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    
    total_acc = 0.0
    success_count = 0
    
    for dataset_name, acc in results.items():
        print(f"{dataset_name}: {acc:.4f}")
        total_acc += acc
        if acc >= 0.92:
            success_count += 1
    
    avg_acc = total_acc / len(results)
    print(f"\n平均准确率: {avg_acc:.4f}")
    print(f"达到92%的数据集数量: {success_count}/{len(results)}")
    
    # 保存结果
    if trainer.config['experiment']['results_file']:
        with open(trainer.config['experiment']['results_file'], 'w') as f:
            f.write('dataset\taccuracy\n')
            for dataset_name, acc in results.items():
                f.write(f"{dataset_name}\t{acc:.4f}\n")

if __name__ == "__main__":
    main()