import numpy as np
import matplotlib.pyplot as plt

# 定义多项式计算函数（系数按x^0到x^n顺序）
def eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))

# 原始数据（从x^0到x^8的系数）
true_coeffs = [3.1683, 10, 5.2863, 2.0853, 2.5811, 1.781, 10, 5.8771, 8.9894]
fitted_coeffs = [362.77162634, 0, 0, 0,
                0, 0, 0, 0, 13.62616131 ]

# 生成数据点
x = np.linspace(0, 10, 500)  # 在[-2, 2]区间生成500个点
y_true = [eval_poly(true_coeffs, xi) for xi in x]
y_fitted = [eval_poly(fitted_coeffs, xi) for xi in x]

# 创建专业级可视化
plt.figure(figsize=(12, 7), dpi=100)
plt.plot(x, y_true,
         color='#2a4d69',
         linewidth=3,
         label='True Polynomial')
plt.plot(x, y_fitted,
         color='#c83e4d',
         linewidth=2.5,
         linestyle='--',
         label='Fitted Polynomial')

# 图表装饰
plt.title('True vs Fitted Polynomial Functions\n',
         fontsize=14, fontweight='bold', color='#2d3436')
plt.xlabel('x value', fontsize=12, labelpad=10)
plt.ylabel('f(x)', fontsize=12, labelpad=10)
plt.grid(True,
         color='#dfe6e9',
         linestyle='--',
         linewidth=0.8,
         alpha=0.8)

# 设置坐标轴范围
plt.xlim(0, 10)
plt.ylim(0, 10000000000)  # 根据实际计算结果调整

# 专业级图例设置
legend = plt.legend(loc='upper left',
                   frameon=True,
                   shadow=True,
                   edgecolor='#2d3436',
                   facecolor='#f8f9fa')
legend.get_frame().set_linewidth(1.2)

# 显示图表
plt.tight_layout()
plt.show()