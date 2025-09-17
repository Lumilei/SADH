import os
import subprocess
import json
import glob

# 参数网格
lambda_recon_grid = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

# 获取所有 Pareto 数据集
pareto_files = glob.glob('handwritten_pareto_missing_*.mat')
pareto_files.sort()  # 按文件名排序

print(f"Found {len(pareto_files)} Pareto datasets:")
for f in pareto_files:
    print(f"  - {f}")

# 结果保存
log_file = 'grid_search_lambda_recon_results.json'
results = []

# 对每个数据集进行网格搜索
for mat_file in pareto_files:
    dataset_name = os.path.basename(mat_file).replace('.mat', '')
    print(f"\n==== Processing dataset: {dataset_name} ====")
    
    dataset_results = []
    
    for lam in lambda_recon_grid:
        print(f"\n---- Running with lambda_recon={lam} on {dataset_name} ----")
        
        cmd = [
            'python', '../../single_experimrnt_entry.py',
            '--lambda_recon', str(lam),
            '--B', '50',
            '--k', '5', 
            '--theta_U', '0.5',
            '--beta', '0.9',
            '--log_file', f'log_{dataset_name}_{lam}.txt',
            mat_file
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # 运行实验
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        
        # 提取最终准确率
        acc = None
        for line in output.splitlines():
            if 'Final Test Accuracy' in line:
                try:
                    acc = float(line.strip().split()[-1])
                except Exception:
                    acc = None
                break
        
        result = {
            'dataset': dataset_name,
            'lambda_recon': lam,
            'accuracy': acc,
            'log_file': f'log_{dataset_name}_{lam}.txt'
        }
        
        dataset_results.append(result)
        results.append(result)
        
        print(f"Dataset: {dataset_name}, lambda_recon={lam}, accuracy={acc}")
        
        # 保存每次的完整输出
        with open(f'output_{dataset_name}_{lam}.txt', 'w', encoding='utf-8') as f:
            f.write(output)

# 保存所有结果
with open(log_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"\n==== Grid search finished! Results saved to: {log_file} ====")

# 打印汇总结果
print("\n==== Summary ====")
for dataset_name in set(r['dataset'] for r in results):
    dataset_results = [r for r in results if r['dataset'] == dataset_name]
    best_result = max(dataset_results, key=lambda x: x['accuracy'] if x['accuracy'] is not None else 0)
    print(f"{dataset_name}: Best accuracy = {best_result['accuracy']:.4f} (lambda_recon={best_result['lambda_recon']})")
