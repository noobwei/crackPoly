import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from time import time
from matplotlib.ticker import ScalarFormatter
from numpy.polynomial import Polynomial


def normalized_poly(x, coeffs):
    """使用标准多项式类进行求值"""
    return Polynomial(coeffs)(x)


def fit_polynomial(y_data, max_degree=8, precision_level=2):
    """
    高精度正系数多项式拟合
    参数：
    - max_degree: 最大多项式阶数 (默认8)
    - precision_level: 精度控制级别 (1-3, 越高越精确)
    """
    results = []
    sorted_y = np.sort(y_data)

    # 精度控制参数
    precision_params = {
        1: {'maxiter': 500, 'ftol': 1e-4, 'reg': 1e-4},  # 快速模式
        2: {'maxiter': 2000, 'ftol': 1e-6, 'reg': 1e-6},  # 标准模式（默认）
        3: {'maxiter': 5000, 'ftol': 1e-8, 'reg': 1e-8}  # 高精度模式
    }
    params = precision_params.get(precision_level, precision_params[2])

    # Stage 1: 自动范围估计
    print("\n🔍 Stage 1: Estimating unified x-range...")
    initial_guess = [np.percentile(y_data, 5), np.percentile(y_data, 95)]
    res = minimize(lambda p: np.mean((np.linspace(p[0], p[1], len(y_data)) - sorted_y)  ** 2),
    initial_guess, method = 'L-BFGS-B',
                            bounds = [(min(y_data), max(y_data))] * 2)
    a_uni, b_uni = res.x
    print(f"✅ Unified x-range: [{a_uni:.2f}, {b_uni:.2f}]\n")

    # Stage 2: 带约束的优化拟合
    print(f"🔧 Stage 2: Fitting up to degree {max_degree} (Precision Level: {precision_level})...")
    x_unified = np.linspace(a_uni, b_uni, len(y_data))

    # 数据标准化
    x_scaled = (x_unified - a_uni) / (b_uni - a_uni)
    y_min, y_max = sorted_y.min(), sorted_y.max()
    y_scale = y_max - y_min if y_max != y_min else 1.0
    y_scaled = (sorted_y - y_min) / y_scale

    for degree in range(1, max_degree + 1):
        start = time()

        # 智能初始化（带随机扰动）
        np.random.seed(42)
        initial_guess = np.array([1 / (10 ** i) + np.random.normal(0, 0.01)
                                  for i in range(degree + 1)])

        # 损失函数（带自适应正则化）
        def loss(coeffs):
            try:
                poly = Polynomial(coeffs)
                y_pred = poly(x_scaled) * y_scale + y_min
                mse = np.mean((y_pred - sorted_y) ** 2)

                # 自适应正则化（基于系数阶数）
                reg = params['reg'] * np.sum(coeffs ** 2 * np.arange(1, degree + 2) ** 2)
                return mse + reg
            except:
                return np.inf

        # 优化器配置
        bounds = [(1e-10, None)] * (degree + 1)
        options = {
            'maxiter': params['maxiter'],
            'ftol': params['ftol'],
            'gtol': 1e-12,
            'disp': False
        }

        res = minimize(loss, initial_guess, method='L-BFGS-B',
                       bounds=bounds, options=options)

        # 结果后处理
        if res.success:
            # 逆标准化系数
            scale_factors = np.array([(b_uni - a_uni) ** i / y_scale
                                      for i in range(degree + 1)])
            final_coeffs = res.x * scale_factors
        else:
            print(f"⚠️ Degree {degree} 优化失败: {res.message}")
            final_coeffs = np.full(degree + 1, np.nan)

        # 存储结果
        results.append({
            'degree': degree,
            'coeffs': final_coeffs.tolist(),
            'time': time() - start,
            'mse': res.fun,
            'x_range': (a_uni, b_uni)
        })

        # 详细系数输出
        print(f"\nDegree {degree} Coefficients:")
        exp_labels = [f"x^{degree - i}" for i in range(degree + 1)]
        for i, (label, coeff) in enumerate(zip(exp_labels, final_coeffs)):
            print(f"{label}: {coeff:.4e}", end='  ')
            if (i + 1) % 3 == 0 or i == degree:
                print()
        print(f"MSE: {results[-1]['mse']:.2e} | Time: {results[-1]['time']:.2f}s")

    return results


def generate_analysis_plots(results, original_data_path):
    # """增强型可视化"""
    # df = pd.read_csv(original_data_path, comment='#')
    # x_real = df['x'].values
    # y_real = df['y'].values
    #
    # plt.figure(figsize=(18, 6))
    #
    # # 拟合曲线对比
    # plt.subplot(1, 3, 1)
    # a, b = results[0]['x_range']
    # x_plot = np.linspace(a, b, 1000)
    #
    # # 原始数据归一化
    # x_norm = np.interp(x_real, (x_real.min(), x_real.max()), (a, b))
    # plt.scatter(x_norm, y_real, s=15, alpha=0.6, color='red', label='Original Data')
    #
    # # 绘制拟合曲线
    # colors = plt.cm.jet(np.linspace(0, 1, len(results)))
    # for res, c in zip(results, colors):
    #     if np.isnan(res['coeffs']).any():
    #         continue
    #     y_curve = normalized_poly(x_plot, res['coeffs'])
    #     plt.plot(x_plot, y_curve, color=c, lw=1.5,
    #              label=f'Degree {res["degree"]}\nMSE: {res["mse"]:.1e}')
    # plt.title('Polynomial Fitting Comparison')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 误差分析
    plt.subplot(1, 3, 2)
    degrees = [r['degree'] for r in results]
    mse_values = [r['mse'] for r in results]
    plt.semilogy(degrees, mse_values, 's-', markersize=8)
    plt.title('MSE vs Polynomial Degree')
    plt.xlabel('Degree')
    plt.ylabel('Log MSE')
    plt.grid(True, alpha=0.3)

    # 时间分析
    plt.subplot(1, 3, 3)
    times = [r['time'] for r in results]
    plt.plot(degrees, times, 'o--', color='darkgreen')
    plt.title('Computation Time Analysis')
    plt.xlabel('Degree')
    plt.ylabel('Time (s)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hackPoly1.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    # 使用示例
    data = pd.read_csv('polynomial_data.csv', comment='#')

    # 参数配置
    config = {
        'max_degree': 10,  # 尝试的最大阶数
        'precision_level': 3  # 1-快速, 2-标准, 3-高精度
    }

    results = fit_polynomial(data['y'].values,  ** config)
    generate_analysis_plots(results, 'polynomial_data.csv')
    print("\n✅ 分析完成！结果保存至 hackPoly1.png")