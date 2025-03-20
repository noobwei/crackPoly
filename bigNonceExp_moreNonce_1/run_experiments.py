import subprocess

# 实验配置
EXPERIMENT_SETTINGS = [
    {'nonce_range': (0, 100), 'samples': [5, 10, 20]},
    {'nonce_range': (0, 200), 'samples': [5, 10, 20]},
    {'nonce_range': (0, 500), 'samples': [5, 10, 20]},
    {'nonce_range': (0, 1000), 'samples': [5, 10, 20]},
    {'nonce_range': (0, 2000), 'samples': [5, 10, 20]}
]

def run_experiment(nonce_min: float, nonce_max: float):
    # 生成数据文件名模板
    data_file = f"data_{nonce_min}_{nonce_max}.csv"
    results_file = f"results_{nonce_min}_{nonce_max}.csv"
    summary_file = f"summary_{nonce_min}_{nonce_max}.csv"

    # 步骤1: 生成数据集
    subprocess.run([
        'python', 'DataGen.py',
        '--nonce_min', str(nonce_min),
        '--nonce_max', str(nonce_max),
        '-o', data_file
    ], check=True)

    # 步骤2: 运行多项式拟合实验
    subprocess.run([
        'python', 'hackPoly.py',
        '--data', data_file,
        '-o', results_file
    ], check=True)

    # 步骤3: 验证实验结果
    subprocess.run([
        'python', 'hackVerifier.py',
        '--data', data_file,
        '--results', results_file,
        '--nonce_min', str(nonce_min),  # 新增参数
        '--nonce_max', str(nonce_max),  # 新增参数
        '-o', summary_file
    ], check=True)

    print(f"\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ EXPERIMENT COMPLETE ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    print(f"Nonce范围: {nonce_min}-{nonce_max}")
    print(f"数据文件: {data_file}")
    print(f"结果文件: {results_file}")
    print(f"总结文件: {summary_file}\n")

if __name__ == "__main__":
    for setting in EXPERIMENT_SETTINGS:
        nonce_min, nonce_max = setting['nonce_range']
        run_experiment(nonce_min, nonce_max)