import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from time import time
from matplotlib.ticker import ScalarFormatter
from numpy.polynomial import Polynomial
import re

# 新增配置参数
SAMPLE_SIZE = 200  # 可在此修改随机抽样数量 [用户可调节参数]
RANDOM_SEED = 42  # 固定随机种子保证可重复性


def normalized_poly(x, coeffs):
    """使用标准Polynomial类进行求值"""
    return Polynomial(coeffs)(x)


def fit_polynomial(y_data, max_degree=8):
    """带正系数约束和数值稳定的多项式拟合"""
    results = []
    sorted_y = np.sort(y_data)

    # Stage 1: 估计统一范围
    print("\n🔍 Stage 1: Estimating unified x-range...")
    initial_guess = [np.percentile(y_data, 5), np.percentile(y_data, 95)]
    res = minimize(lambda p: np.mean((np.linspace(p[0], p[1], len(y_data)) - sorted_y)   ** 2),
    initial_guess, method = 'L-BFGS-B',
                            bounds = [(min(y_data), max(y_data))] * 2)
    a_uni, b_uni = res.x
    print(f"✅ Unified x-range: [{a_uni:.2f}, {b_uni:.2f}]\n")

    # Stage 2: 改进的带约束拟合
    print("🔧 Stage 2: Fitting polynomials with positive coefficients...")
    x_unified = np.linspace(a_uni, b_uni, len(y_data))

    # 数据归一化
    x_scaled = (x_unified - a_uni) / (b_uni - a_uni)  # 缩放到[0,1]
    y_min, y_max = sorted_y.min(), sorted_y.max()
    y_scaled = (sorted_y - y_min) / (y_max - y_min)  # 缩放到[0,1]

    for degree in range(1, max_degree + 1):
        start = time()

        # 改进的初始猜测（指数衰减）
        initial_guess = np.array([1 / (10 ** i) for i in range(degree + 1)])

        # 损失函数（带L2正则化）
        def loss(coeffs):
            try:
                poly = Polynomial(coeffs)
                y_pred = poly(x_scaled)  # 在归一化数据上计算
                y_pred = y_pred * (y_max - y_min) + y_min  # 逆缩放
                mse = np.mean((y_pred - sorted_y) ** 2)
                reg = 1e-6 * np.sum(coeffs  ** 2)  # 正则化项
                return mse + reg
            except:
                return np.inf

        # 带约束优化
        bounds = [(1e-10, None)] * (degree + 1)
        res = minimize(loss, initial_guess, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 2000})

        # 检查收敛性
        if not res.success:
            print(f"⚠️ Degree {degree} 未收敛: {res.message}")

        # 格式化系数（考虑归一化缩放）
        final_coeffs = res.x * (y_max - y_min) / (np.array([(b_uni - a_uni)  ** i for i in range(degree + 1)]))

        # 存储结果
        results.append({
            'degree': degree,
            'coeffs': final_coeffs.tolist(),
            'time': time() - start,
            'mse': res.fun,
            'x_range': (a_uni, b_uni)
        })

        # 输出系数详情
        print(f"\nDegree {degree} Coefficients (all > 0):")
        formatted_coeffs = [f"{c:.4e}" for c in final_coeffs]
        for i, (exp, c) in enumerate(zip(range(degree, -1, -1), formatted_coeffs)):
            print(f"  x^{exp}: {c}", end='  ')
            if (i + 1) % 3 == 0:  # 每行显示3个系数
                print()
        print(f"\n  MSE: {results[-1]['mse']:.2e} | Time: {results[-1]['time']:.2f}s")

    return results


def generate_analysis_plots(results, original_data_path, true_coeffs):
    """增强可视化：新增参数对比子图"""
    # 加载原始数据
    df = pd.read_csv(original_data_path, comment='#')
    x_real = df['x'].values
    y_real = df['y'].values

    plt.figure(figsize=(18, 6))

    # 子图1：统一范围比较
    plt.subplot(1, 3, 1)
    a, b = results[0]['x_range']
    x_normalized = np.interp(x_real, (x_real.min(), x_real.max()), (a, b))
    plt.scatter(x_normalized, y_real, s=15, alpha=0.6, color='red', label='Original Data')
    x_plot = np.linspace(a, b, 1000)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for res, c in zip(results, colors):
        y_curve = normalized_poly(x_plot, res['coeffs'])
        plt.plot(x_plot, y_curve, color=c, lw=2, label=f'Deg {res["degree"]} (MSE:{res["mse"]:.1e})')
    plt.xlabel('Unified X'), plt.ylabel('Y'), plt.title('Function Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'), plt.grid(alpha=0.2)

    # 子图2：时间分析
    plt.subplot(1, 3, 2)
    degrees = [r['degree'] for r in results]
    plt.plot(degrees, [r['time'] for r in results], 's-', markersize=8, color='darkorange', lw=2)
    plt.xlabel('Degree'), plt.ylabel('Time (s)'), plt.title('Time Complexity')
    plt.yscale('log'), plt.grid(True), plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    # 新增子图3：参数误差分析
    plt.subplot(1, 3, 3)
    coeff_errors = []
    true_degree = len(true_coeffs) - 1
    for res in results:
        fit_degree = res['degree']
        # 对齐系数长度
        max_len = max(len(res['coeffs']), len(true_coeffs))
        fit_coeffs = np.pad(res['coeffs'], (0, max_len - len(res['coeffs'])), 'constant')
        true_padded = np.pad(true_coeffs, (0, max_len - len(true_coeffs)), 'constant')
        # 计算匹配误差
        error = np.sqrt(np.mean((fit_coeffs - true_padded) ** 2))
        coeff_errors.append(error)

    plt.semilogy(degrees, coeff_errors, 'D-', color='purple', markersize=8)
    plt.axvline(true_degree, color='gray', linestyle='--', label=f'True Degree ({true_degree})')
    plt.xlabel('Polynomial Degree'), plt.ylabel('RMS Coefficient Error')
    plt.title('Parameter Accuracy Analysis'), plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('hackPoly2.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    # 读取真实系数
    with open('polynomial_data.csv', 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('#'):
            # 提取多项式表达式
            content = first_line.lstrip('# ').split(' = ')[1]  # 获取等号右侧表达式
            # 使用正则表达式提取系数
            coeffs = re.findall(r'([+-]?\d+\.\d+)x\^\d+', content)
            true_coeffs = [float(c) for c in coeffs]
            print(f"\n🔍 真实参数: {true_coeffs}")
        else:
            raise ValueError("CSV文件中未找到真实参数")
    # 加载并抽样数据
    df = pd.read_csv('polynomial_data.csv', comment='#')
    df_sampled = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)
    y_data = df_sampled['y'].values
    print(f"\n📊 使用{SAMPLE_SIZE}个样本进行拟合 (随机种子={RANDOM_SEED})")

    # 运行拟合
    fit_results = fit_polynomial(y_data, max_degree=10)

    # 参数对比分析
    print("\n🔬 参数对比结果:")
    true_degree = len(true_coeffs) - 1
    for res in fit_results:
        fit_degree = res['degree']
        # 对齐系数长度
        max_len = max(len(res['coeffs']), len(true_coeffs))
        fit_coeffs = np.pad(res['coeffs'], (0, max_len - len(res['coeffs'])), 'constant')
        true_padded = np.pad(true_coeffs, (0, max_len - len(true_coeffs)), 'constant')
        # 计算误差指标
        mse = np.mean((fit_coeffs - true_padded) ** 2)
        mae = np.mean(np.abs(fit_coeffs - true_padded))
        # 特殊标注匹配度
        match_note = "(匹配真实阶数)" if fit_degree == true_degree else ""
        print(f"Degree {fit_degree:2d} | MSE: {mse:.2e}  MAE: {mae:.2e} {match_note}")

    # 生成图形
    generate_analysis_plots(fit_results, 'polynomial_data.csv', true_coeffs)
    print("\n✅ 分析完成! 结果保存至:")
    print("hackPoly2.png")