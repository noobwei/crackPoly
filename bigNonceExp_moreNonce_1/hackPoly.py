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
    import argparse
    import os

    # 配置命令行参数
    parser = argparse.ArgumentParser(description='Polynomial Fitting Experiments')
    parser.add_argument('--data', type=str, default="polynomial_data.csv", help='输入数据文件路径')
    parser.add_argument('-o', '--output', type=str, default="experiment_results.csv", help='实验结果输出路径')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output) or '.'  # 处理空路径情况
    os.makedirs(output_dir, exist_ok=True)

    # 实验配置
    SAMPLE_SIZES = [5, 10, 20]
    NUM_EXPERIMENTS = 100

    # 保存实验结果（使用参数中的输出路径）
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_size', 'experiment_id'] + [f'coeff_{i}' for i in range(9)])

        print(f"▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ EXPERIMENT START ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
        print(f"Input data: {args.data}")
        print(f"Output file: {args.output}")

        for sample_size in SAMPLE_SIZES:
            print(f"\n▶▶ 正在处理样本量 {sample_size}...")

            for exp_id in range(NUM_EXPERIMENTS):
                # 随机采样数据
                try:
                    x_data, y_data = load_dataset(args.data, sample_size)
                except FileNotFoundError:
                    print(f"错误：数据文件 {args.data} 不存在")
                    exit(1)

                # 执行多项式拟合
                coeffs = fit_polynomial(x_data, y_data)

                # 记录结果
                if coeffs:  # 仅保存有效结果
                    writer.writerow([sample_size, exp_id] + coeffs)

                # 实时进度显示
                if (exp_id + 1) % 10 == 0:
                    print(f"样本量 {sample_size} - 已完成 {exp_id + 1}/{NUM_EXPERIMENTS} 次实验")

    print("\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ EXPERIMENT COMPLETE ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    print(f"✅ 实验结果已保存至 {args.output}")