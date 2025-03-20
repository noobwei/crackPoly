import csv
import numpy as np
from scipy.optimize import bisect


def inverse_calculation(coeffs: list, data_path: str) -> float:
    """计算单次实验的阈值命中率"""
    threshold = 5e-4
    success_count = 0
    total_count = 0

    def poly_func(x):
        return sum(c * (x  ** i) for i, c in enumerate(coeffs))

    def find_root(y_target, nonce):
        try:
            # 动态计算搜索范围 ±0.2
            search_min = 2 + nonce - 0.2
            search_max = 2 + nonce + 0.2
            return bisect(lambda x: poly_func(x) - y_target,
                          search_min, search_max, xtol=1e-6)  # 放宽精度要求
        except:
            return None

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for row in reader:
            x_true = float(row[0])
            nonce = float(row[1])
            y = float(row[2])

            root = find_root(y, nonce)
            if root is None:
                continue

            x_pred = root - nonce
            if abs(x_pred - x_true) <= threshold:
                success_count += 1
            total_count += 1

    return success_count / total_count if total_count > 0 else 0.0


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    # 配置命令行参数
    parser = argparse.ArgumentParser(description='Polynomial Experiment Verifier')
    parser.add_argument('--data', type=str, required=True, help='Input data file')
    parser.add_argument('--results', type=str, required=True, help='Experiment results file')
    parser.add_argument('--nonce_min', type=float, required=True, help='Minimum nonce value')
    parser.add_argument('--nonce_max', type=float, required=True, help='Maximum nonce value')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output summary file')
    args = parser.parse_args()

    print("实验开始")

    # 读取实验结果
    experiment_data = defaultdict(list)
    try:
        with open(args.results, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                sample_size = int(row[0])
                if sample_size in [5, 10, 20]:
                    coeffs = list(map(float, row[2:11]))
                    experiment_data[sample_size].append(coeffs)
    except FileNotFoundError:
        print(f"错误：结果文件 {args.results} 不存在")
        exit(1)

    # 计算平均命中率
    summary = []
    TARGET_SAMPLES = [5, 10, 20]
    for sample_size in TARGET_SAMPLES:
        hit_rates = []
        for coeffs in experiment_data.get(sample_size, []):
            hr = inverse_calculation(coeffs, args.data)
            hit_rates.append(hr)

        avg_hr = np.mean(hit_rates) if hit_rates else 0.0
        std_hr = np.std(hit_rates) if hit_rates else 0.0
        summary.append([
            args.nonce_min,    # 记录 nonce 范围
            args.nonce_max,    # 记录 nonce 范围
            sample_size,
            avg_hr,
            std_hr,
            np.min(hit_rates) if hit_rates else 0.0,
            np.max(hit_rates) if hit_rates else 0.0
        ])

    # 保存结果（包含 nonce 范围信息）
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'NonceMin', 'NonceMax', 'SampleSize',
            'AvgHitRate', 'StdDev',
            'MinHitRate', 'MaxHitRate'
        ])
        writer.writerows(summary)

    # 打印增强版结果
    print("\n▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ VERIFICATION SUMMARY ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    print(f"Nonce 范围: {args.nonce_min} ~ {args.nonce_max}")
    print(f"验证数据: {args.data}")
    print(f"分析结果: {args.results}")
    print("\nNonce范围 | 样本量 | 平均命中率 | 标准差 | 最小命中率 | 最大命中率")
    print("--------------------------------------------------------------------")
    for row in summary:
        print(f"{row[0]:<7.1f}-{row[1]:<5.1f} | {row[2]:^6d} | {row[3]:>10.2%} | {row[4]:.4f} | {row[5]:>10.2%} | {row[6]:>10.2%}")
    print(f"\n✅ 验证结果已保存至 {args.output}")