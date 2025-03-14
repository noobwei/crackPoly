import numpy as np

# 加载8次多项式系数（从x^0到x^8）
COEFFS = np.loadtxt('coeff_8th.csv').tolist()

def calculate_polynomial(x: float) -> float:
    """使用霍纳法高效计算多项式值"""
    result = 0.0
    # 反向遍历系数（从高次项到低次项）
    for coeff in reversed(COEFFS):
        result = result * x + coeff
    return result

if __name__ == "__main__":
    # 支持高精度输入格式
    try:
        x = float(input("请输入高精度浮点数 x: "))
        result = calculate_polynomial(x)
        # 输出保留15位小数
        print(f"\n多项式计算结果: {result:.15f}")
        print("计算项分解:")
        # 显示各次项贡献值
        for power, coeff in enumerate(COEFFS):
            term = coeff * (x ** power)
            print(f"x^{power}: {term:+.15e}")
    except ValueError:
        print("错误：请输入有效的浮点数")
    except FileNotFoundError:
        print("错误：未找到coefficients.csv文件")