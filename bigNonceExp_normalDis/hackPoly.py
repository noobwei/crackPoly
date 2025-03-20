import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def parse_original_polynomial(filename: str) -> List[float]:
    """从CSV文件头解析原始多项式系数"""
    with open(filename, 'r') as f:
        line = f.readline().strip()
        equation = line.split(": ")[1]

    pattern = re.compile(r"([+-]?[\d\.]+)x\^(\d+)")
    terms = pattern.findall(equation.replace(' ', ''))

    coeff_dict = {}
    for coeff, power in terms:
        coeff = float(coeff)
        power = int(power)
        coeff_dict[power] = coeff

    max_power = max(coeff_dict.keys())
    return [coeff_dict.get(i, 0.0) for i in range(max_power + 1)]


def load_dataset(filename: str, num_samples: int = None) -> tuple:
    """加载数据集并随机选取指定数量的样本"""
    x_values, y_values = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过注释行
        next(reader)  # 跳过标题行
        rows = list(reader)

        if num_samples is not None and len(rows) >= num_samples:
            indices = np.random.choice(len(rows), size=num_samples, replace=False)
            selected_rows = [rows[i] for i in indices]
        else:
            selected_rows = rows

        for row in selected_rows:
            nonce = float(row[1])
            y = float(row[2])
            x_values.append(2 + nonce)
            y_values.append(y)
    return np.array(x_values), np.array(y_values)


def fit_polynomial(x: np.ndarray, y: np.ndarray) -> List[float]:
    """进行8次多项式拟合"""
    try:
        # 执行多项式拟合（自动包含x^0到x^8）
        coeffs = np.polyfit(x, y, deg=8)
        # 反转系数顺序为升幂排列
        return coeffs[::-1].tolist()
    except Exception as e:
        print(f"拟合失败: {str(e)}")
        return []


def get_8th_degree_coeffs(results_dict: dict) -> List[float]:
    """提取完整的8次多项式系数"""
    return results_dict.get(8, [0.0] * 9)[:9]


if __name__ == "__main__":
    # 配置参数
    DATA_FILE = "polynomial_data.csv"
    SAMPLE_SIZES = [2, 5, 10, 20, 50, 100]
    NUM_EXPERIMENTS = 100

    # 保存实验结果
    with open('experiment_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['sample_size', 'experiment_id'] + [f'coeff_{i}' for i in range(9)]
        writer.writerow(header)

        for sample_size in SAMPLE_SIZES:
            for exp_id in range(NUM_EXPERIMENTS):
                # 随机采样数据
                x_data, y_data = load_dataset(DATA_FILE, sample_size)

                # 执行8次多项式拟合
                coeffs = fit_polynomial(x_data, y_data)

                # 记录结果（自动包含9个系数）
                if coeffs:  # 仅记录成功拟合的结果
                    writer.writerow([sample_size, exp_id] + coeffs)
                print(f'Sample: {sample_size}, Exp: {exp_id} - 系数数量: {len(coeffs)}')