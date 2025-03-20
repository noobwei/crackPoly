import random
import csv
from typing import List, Tuple


def generate_float_number(max_integer: int,
                          decimal_bits: int,
                          allow_negative: bool = False) -> float:
    """Generate floating-point number between 0 and max_integer with specified precision"""
    integer_part = random.randint(0, max_integer)
    if integer_part == max_integer:
        decimal_part = 0
    else:
        decimal_part = random.randint(0, (10   ** decimal_bits) - 1) / 10   ** decimal_bits
    number = integer_part + decimal_part
    if allow_negative and random.choice([True, False]):
        number *= -1
    return round(number, decimal_bits)


def generate_polynomial_coefficients(degree: int,
                                     max_integer: int,
                                     decimal_bits: int) -> List[float]:
    """Generate polynomial coefficients between 0 and max_integer"""
    return [generate_float_number(max_integer, decimal_bits, False)
            for _ in range(degree + 1)]


def calculate_y(x: float,
                coefficients: List[float],
                y_decimal: int) -> float:
    """Calculate polynomial value at x"""
    return round(sum(c * (x   ** i) for i, c in enumerate(coefficients)), y_decimal)


def generate_dataset(
        coefficients: List[float],
        num_samples: int,
        y_decimal: int,
        nonce_decimal: int,
        nonce_min: float,
        nonce_max: float
) -> List[Tuple[float, float, float]]:
    """Generate dataset with x_base, nonce and f(x_base+nonce)"""
    dataset = []

    # 正态分布参数
    mu = 2.0  # 均值
    sigma = 0.0333  # 标准差 (覆盖99.7%的1.9-2.1范围)

    for _ in range(num_samples):
        # 生成正态分布的x_base（强制限定在1.9-2.1之间）
        while True:
            x_base = random.gauss(mu, sigma)
            if 1.9 <= x_base <= 2.1:
                x_base = round(x_base, 8)  # 保留8位小数
                break

        nonce = round(random.uniform(nonce_min, nonce_max), nonce_decimal)
        x_combined = x_base + nonce
        y = calculate_y(x_combined, coefficients, y_decimal)

        dataset.append((x_base, nonce, y))
    return dataset


def save_to_csv(data: List[Tuple[float, float, float]],
                filename: str,
                equation: str,
                x_dec: int,
                nonce_dec: int,
                y_dec: int) -> None:
    """Save data with x_base, nonce and f(x+nonce)"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Polynomial: {equation}"])
        writer.writerow(['x_base', 'nonce', 'f(x_base+nonce)'])
        for x, n, y in data:
            writer.writerow([
                f"{x:.{x_dec}f}",
                f"{n:.{nonce_dec}f}",
                f"{y:.{y_dec}f}"
            ])


if __name__ == "__main__":
    import argparse

    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--nonce_min', type=float, default=100.0)
    parser.add_argument('--nonce_max', type=float, default=200.0)
    parser.add_argument('-o', '--output', type=str, default="polynomial_data.csv")
    args = parser.parse_args()

    POLY_DEGREE = 8
    X_DEC_BITS = 8
    NONCE_DEC_BITS = 4
    Y_DECIMAL = 5
    NUM_SAMPLES = 100000
    COEFF_MAX_INTEGER = 10
    COEFF_DEC_BITS = 4

    coefficients = generate_polynomial_coefficients(POLY_DEGREE, COEFF_MAX_INTEGER, COEFF_DEC_BITS)
    equation = "y = " + " + ".join([f"{c:.{COEFF_DEC_BITS}f}x^{i}" for i, c in enumerate(coefficients)])

    # 生成数据集时传入 nonce 范围
    dataset = generate_dataset(coefficients, NUM_SAMPLES, Y_DECIMAL, NONCE_DEC_BITS,
                               args.nonce_min, args.nonce_max)

    save_to_csv(dataset, args.output, equation, X_DEC_BITS, NONCE_DEC_BITS, Y_DECIMAL)
    print(f"▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ DATA GENERATED ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    print(f"✅ 数据已保存至 {args.output}")
    print(f"Nonce范围: {args.nonce_min} ~ {args.nonce_max}")