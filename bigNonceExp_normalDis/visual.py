import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_success_rates():
    # 原始数据直接嵌入
    raw_data = """
    Sample size: 2, Average hit rate: 0.09%
    Sample size: 5, Average hit rate: 0.27%
    Sample size: 10, Average hit rate: 0.32%
    Sample size: 20, Average hit rate: 0.60%
    Sample size: 50, Average hit rate: 1.28%
    Sample size: 100, Average hit rate: 1.88%
    """

    # 数据解析（保留原始百分比值）
    sample_sizes = []
    success_rates = []
    for line in raw_data.strip().split('\n'):
        parts = line.strip().split(',')
        size = int(parts[0].split(': ')[1])
        rate = float(parts[1].split(': ')[1].replace('%', ''))  # 直接使用百分比数值
        sample_sizes.append(size)
        success_rates.append(rate)

    # 创建专业级微调图表
    plt.figure(figsize=(8, 4.5), dpi=150)
    ax = plt.gca()

    # 绘制优化后的组合图
    ax.plot(sample_sizes, success_rates,
            marker='o',
            markersize=5,  # 缩小标记尺寸
            linestyle='--',  # 改为虚线连接
            linewidth=1.2,
            color='#2c7bb6',
            markerfacecolor='#d7191c',
            markeredgecolor='black',
            markeredgewidth=0.5)

    # 坐标轴精细化设置
    ax.set_xlabel('Sample Size', fontsize=11, labelpad=8)
    ax.set_ylabel('Success Rate (%)', fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=2))  # 原生百分比格式化

    # 网格和刻度优化
    ax.set_xticks(sample_sizes)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 针对低值优化显示范围
    ax.set_ylim(-0.05, 2.0)  # 显示0-2%范围
    ax.set_xlim(0, 105)  # 留出右侧空白

    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig('low_success_analysis.svg', format='svg')  # 矢量图格式
    plt.show()


if __name__ == "__main__":
    plot_success_rates()