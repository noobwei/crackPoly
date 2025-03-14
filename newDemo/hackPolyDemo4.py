import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import minimize


def load_data_and_coefficients(filename):
    """Load data and extract true coefficients from comment"""
    with open(filename, 'r') as f:
        header = f.readline().strip()
    matches = re.findall(r"(-?\d+\.\d+)x\^\d+", header)  # 允许负系数匹配
    true_coeffs = [float(m) for m in matches]
    return pd.read_csv(filename, comment='#'), true_coeffs


def format_coefficient(c):
    """Format coefficient with sign awareness"""
    return f"{c:+.8f}".rstrip('0').rstrip('.') if abs(c) > 1e-8 else "0.0"


def run_experiment_with_visualization():
    # Load data and true coefficients
    full_data, true_coeffs = load_data_and_coefficients('polynomial_data.csv')

    # Randomly select 10 samples
    np.random.seed(42)
    train_data = full_data.sample(n=10, random_state=42)
    sorted_data = train_data.sort_values('y_base').reset_index(drop=True)

    # 关键修改1：使用实际x值进行拟合
    x_real = sorted_data['x_base'].values
    y_real = sorted_data['y_base'].values

    # 关键修改2：移除不合理的系数约束
    bounds = [(None, None) for _ in range(9)]  # 允许正负系数

    # 关键修改3：减少约束点数量
    constraints = []
    constraint_point = sorted_data.iloc[[0]]  # 只保留一个约束点
    for _, row in constraint_point.iterrows():
        constraints.append({
            'type': 'eq',
            'fun': lambda c, x=row['x_base'], y=row['y_base']: sum(c[i] * (x ** i) for i in range(9)) - y
        })

    try:
        # 关键修改4：使用实际x值进行优化
        def objective(coeffs):
            predictions = np.array([sum(c * (x ** i) for i, c in enumerate(coeffs)) for x in x_real])
            return np.mean((predictions - y_real) ** 2) + 1e-6 * np.sum(np.square(coeffs))  # 添加正则化

        # 关键修改5：改进初始猜测
        initial_guess = [0.0 if i != 0 else 0.1 for i in range(9)]

        # 运行优化
        res = minimize(objective, initial_guess,
                       method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 5000})

        if not res.success:
            raise RuntimeError(res.message)
        fitted_coeffs = res.x

        # 打印系数对比
        print("=== Coefficient Comparison ===")
        print(f"{'Term':<6} {'True':<16} {'Fitted':<16}")
        for i, (tc, fc) in enumerate(zip(true_coeffs, fitted_coeffs)):
            print(f"x^{i:<2} {format_coefficient(tc):<16} {format_coefficient(fc):<16}")

        # 生成预测曲线
        x_plot = np.linspace(1.9, 2.1, 500)
        true_curve = [sum(c * (x ** i) for i, c in enumerate(true_coeffs)) for x in x_plot]
        fitted_curve = [sum(c * (x ** i) for i, c in enumerate(fitted_coeffs)) for x in x_plot]

        # 可视化
        plt.figure(figsize=(15, 9))
        plt.plot(x_plot, true_curve, 'g--', label='True Curve', alpha=0.7)
        plt.plot(x_plot, fitted_curve, 'r-', label='Fitted Curve')

        # 绘制所有数据点（包括未参与训练的）
        all_x = full_data['x_base']
        all_y = full_data['y_base']
        plt.scatter(all_x, all_y, c='blue', s=10, alpha=0.3, label='All Data Points')

        # 高亮训练点
        plt.scatter(x_real, y_real, c='black', s=50, marker='o',
                    edgecolors='red', label='Training Points')

        plt.xlabel('X Value')
        plt.ylabel('Y Value')
        plt.title('True vs Fitted Polynomial (with Real Data Distribution)')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.savefig('correct_comparison.png', dpi=300)
        print("\nVisualization saved to correct_comparison.png")

        # 计算真实准确率（在整个数据集上）
        full_pred = full_data['x_base'].apply(lambda x: sum(c * (x ** i) for i, c in enumerate(fitted_coeffs)))
        accuracy = ((full_pred >= full_data['y_minus']) & (full_pred <= full_data['y_plus'])).mean()
        print(f"\nGlobal Accuracy: {accuracy * 100:.1f}%")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")


if __name__ == "__main__":
    run_experiment_with_visualization()