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
        decimal_part = random.randint(0, (10  ** decimal_bits) - 1) / 10  ** decimal_bits
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
    return round(sum(c * (x  ** i) for i, c in enumerate(coefficients)), y_decimal)


def generate_dataset(
        coefficients: List[float],
        num_samples: int,
        y_decimal: int,
        nonce_decimal: int
) -> List[Tuple[float, float, float]]:
    """Generate dataset with x_base, nonce and f(x_base+nonce)"""
    dataset = []

    # 生成范围保持不变 [1.9005, 2.0995]
    min_val = 190050000  # 1.90050000
    max_val = 209950000  # 2.09950000

    for _ in range(num_samples):
        # 生成基准x值（8位小数）
        x_base = random.randint(min_val, max_val) / 10 ** 8

        nonce = round(random.uniform(100.0, 200.0), nonce_decimal)

        # 计算f(x_base + nonce)
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
    # Configuration
    POLY_DEGREE = 8
    X_DEC_BITS = 8  # x_base的小数位数
    NONCE_DEC_BITS = 4  # nonce的小数位数
    Y_DECIMAL = 5  # y值的小数位数
    NUM_SAMPLES = 100000
    COEFF_MAX_INTEGER = 10  # 系数最大整数值
    COEFF_DEC_BITS = 4  # 系数小数位数
    OUTPUT_FILE = "polynomial_data.csv"

    # Generate polynomial
    coefficients = generate_polynomial_coefficients(POLY_DEGREE, COEFF_MAX_INTEGER, COEFF_DEC_BITS)
    equation = "y = " + " + ".join([f"{c:.{COEFF_DEC_BITS}f}x^{i}" for i, c in enumerate(coefficients)])

    # Print parameters
    print("▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ POLYNOMIAL DETAILS ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    print(f"Equation: {equation}")
    print("\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ DATA PARAMETERS ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    print(f"x_base范围: 1.90000000-2.10000000 (±0.1 around mean=2)")
    print(f"x_base精度: {X_DEC_BITS} 位小数")
    print(f"nonce精度: {NONCE_DEC_BITS} 位小数")
    print(f"y值精度: {Y_DECIMAL} 位小数")
    print(f"样本数量: {NUM_SAMPLES}")
    print(f"多项式系数范围: 0 到 {COEFF_MAX_INTEGER} 含 {COEFF_DEC_BITS} 位小数")

    # Generate and save data
    dataset = generate_dataset(coefficients, NUM_SAMPLES, Y_DECIMAL, NONCE_DEC_BITS)
    save_to_csv(dataset, OUTPUT_FILE, equation, X_DEC_BITS, NONCE_DEC_BITS, Y_DECIMAL)
    print(f"\n✅ 数据已保存至 {OUTPUT_FILE}")