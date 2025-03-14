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

    # 使用正则表达式解析多项式项
    pattern = re.compile(r"([+-]?[\d\.]+)x\^(\d+)")
    terms = pattern.findall(equation.replace(' ', ''))

    # 创建系数字典
    coeff_dict = {}
    for coeff, power in terms:
        coeff = float(coeff)
        power = int(power)
        coeff_dict[power] = coeff

    # 转换为有序列表
    max_power = max(coeff_dict.keys())
    return [coeff_dict.get(i, 0.0) for i in range(max_power + 1)]


def load_dataset(filename: str, num_samples: int = None) -> tuple:
    """加载数据集并计算实际x值"""
    x_values, y_values = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过注释行
        next(reader)  # 跳过标题行
        for i, row in enumerate(reader):
            # 添加样本数量限制
            if num_samples is not None and i >= num_samples:
                break
            x_base = float(row[0])
            nonce = float(row[1])
            y = float(row[2])
            x_values.append(2 + nonce)
            y_values.append(y)
    return np.array(x_values), np.array(y_values)


def fit_polynomials(x: np.ndarray, y: np.ndarray, max_degree: int) -> dict:
    """进行不同次数的多项式拟合"""
    results = {}
    for degree in range(1, max_degree + 1):
        # 执行多项式拟合
        coeffs = np.polyfit(x, y, deg=degree)
        # 反转系数顺序 (polyfit返回最高次项在前)
        coeffs = coeffs[::-1].tolist()
        # 补零到10次项
        coeffs += [0.0] * (10 - degree) if degree < 10 else []
        results[degree] = coeffs
        print(f"\n▄▄▄▄▄▄▄▄ 拟合 {degree} 次多项式 ▄▄▄▄▄▄▄▄")
        for i, c in enumerate(coeffs[:degree + 1]):
            print(f"x^{i}: {c:.10f}")
    return results



def plot_comparison(original_coeffs: List[float], fitted_results: dict):
    """绘制原始多项式与拟合结果对比图"""
    plt.figure(figsize=(14, 8))

    # 生成采样点
    x_plot = np.linspace(100, 200, 1000)

    # 绘制原始多项式
    original_y = np.zeros_like(x_plot)
    for i, c in enumerate(original_coeffs):
        original_y += c * x_plot ** i
    plt.plot(x_plot, original_y, 'k-', lw=3, label='Original Polynomial')

    # 绘制拟合结果
    colors = plt.cm.jet(np.linspace(0, 1, 10))
    for degree in range(1, 11):
        coeffs = fitted_results[degree]
        y = np.zeros_like(x_plot)
        for i, c in enumerate(coeffs[:degree + 1]):
            y += c * x_plot ** i
        plt.plot(x_plot, y, color=colors[degree - 1], alpha=0.6,
                 label=f'Degree {degree} Fit')

    plt.scatter(x_data, y_data, c='red', s=100, edgecolors='gold', linewidths=1.5,
                zorder=10,
                marker='o')

    plt.xlabel('x (x_base + nonce)')
    plt.ylabel('y')
    plt.title('Polynomial Comparison (Original vs Fitted)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 配置参数
    DATA_FILE = "polynomial_data.csv"
    MAX_DEGREE = 10
    NUM_SAMPLES = 10

    # 禁用科学计数法显示
    np.set_printoptions(suppress=True)

    # 解析原始多项式
    original_coeffs = parse_original_polynomial(DATA_FILE)
    print("原始多项式系数解析完成")
    print("多项式次数:", len(original_coeffs) - 1)

    # 加载数据集
    x_data, y_data = load_dataset(DATA_FILE, NUM_SAMPLES)
    print(f"\n数据集加载完成，样本数量: {len(x_data)}")

    # 执行多项式拟合
    fitted_results = fit_polynomials(x_data, y_data, MAX_DEGREE)


    def get_8th_degree_coeffs(results_dict: dict) -> List[float]:
        """提取完整的8次多项式系数（包含补零前的实际系数）"""
        # 获取原始拟合结果（含补零到10次的系数）
        raw_coeffs = results_dict[8]
        # 提取实际使用的系数（x^0到x^8）
        actual_coeffs = raw_coeffs[:9]
        return actual_coeffs


    # 获取并打印8次多项式系数
    coefficients_8th = get_8th_degree_coeffs(fitted_results)
    print("\n▄▄▄▄▄▄▄▄▄▄▄▄ 8次多项式完整系数 ▄▄▄▄▄▄▄▄▄▄▄▄")
    for power, coeff in enumerate(coefficients_8th):
        print(f"x^{power}: {coeff:.10f}")
    coefficients_8th = get_8th_degree_coeffs(fitted_results)
    np.savetxt('coeff_8th.csv', coefficients_8th, fmt='%.10f')

    # # 绘制对比图
    # print("\n生成对比图表...")
    plot_comparison(original_coeffs, fitted_results)