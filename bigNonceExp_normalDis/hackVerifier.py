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
            return bisect(lambda x: poly_func(x) - y_target, 1 + nonce, 3 + nonce, xtol=1e-12)
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
    RESULTS_FILE = "experiment_results.csv"
    DATA_FILE = "polynomial_data.csv"

    # 读取实验结果
    experiment_data = {}
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题
        for row in reader:
            sample_size = int(row[0])
            if sample_size not in experiment_data:
                experiment_data[sample_size] = []
            coeffs = list(map(float, row[2:11]))
            experiment_data[sample_size].append(coeffs)

    # 计算平均命中率
    summary = []
    for sample_size in [2, 5, 10, 20, 50, 100]:
        hit_rates = []
        for coeffs in experiment_data.get(sample_size, []):
            hr = inverse_calculation(coeffs, DATA_FILE)
            hit_rates.append(hr)
        avg_hr = np.mean(hit_rates) if hit_rates else 0.0
        summary.append([sample_size, avg_hr])
        print(f'Sample size: {sample_size}, Average hit rate: {avg_hr:.2%}')

    # 保存结果
    with open('final_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SampleSize', 'AverageHitRate'])
        writer.writerows(summary)