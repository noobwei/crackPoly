import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def generate_float_number(max_integer: int,
                          decimal_bits: int,
                          allow_negative: bool = False) -> float:
    """Generate floating-point number between 0 and max_integer with specified precision"""
    integer_part = random.randint(0, max_integer)
    if integer_part == max_integer:
        decimal_part = 0
    else:
        decimal_part = random.randint(0, (10 ** decimal_bits) - 1) / 10 ** decimal_bits
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
    """Calculate polynomial value"""
    return round(sum(c * (x ** i) for i, c in enumerate(coefficients)), y_decimal)


def generate_dataset(
        coefficients: List[float],
        num_samples: int,
        y_decimal: int,
) -> List[Tuple[float, float, float, float]]:
    """Generate dataset with three points per sample"""
    dataset = []
    delta = 0.0005  # 5e-4

    # 生成范围限制在 [1.9005, 2.0995] 保证偏移后不越界
    min_val = 190050000  # 1.90050000
    max_val = 209950000  # 2.09950000

    for _ in range(num_samples):
        # 生成基准x值（小数点后8位）
        x_base = random.randint(min_val, max_val) / 10 ** 8

        # 计算三个y值
        y_original = calculate_y(x_base, coefficients, y_decimal)
        y_plus = calculate_y(round(x_base + delta, 8), coefficients, y_decimal)
        y_minus = calculate_y(round(x_base - delta, 8), coefficients, y_decimal)

        dataset.append((x_base, y_original, y_plus, y_minus))
    return dataset


def save_to_csv(data: List[Tuple[float, float, float, float]],
                filename: str,
                equation: str,
                x_dec: int,
                y_dec: int) -> None:
    """Save data with three y-values per row"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Polynomial: {equation}"])
        writer.writerow(['x_base', 'y_base', 'y_plus', 'y_minus'])
        for x, y, y_p, y_m in data:
            writer.writerow([
                f"{x:.{x_dec}f}",
                f"{y:.{y_dec}f}",
                f"{y_p:.{y_dec}f}",
                f"{y_m:.{y_dec}f}"
            ])


if __name__ == "__main__":
    # Configuration
    POLY_DEGREE = 8
    X_DEC_BITS = 8  # 8 decimal places for x
    Y_DECIMAL = 5  # 5 decimal places for y
    NUM_SAMPLES = 100000
    COEFF_MAX_INTEGER = 10  # 系数最大整数值
    COEFF_DEC_BITS = 4      # 系数小数位数
    OUTPUT_FILE = "polynomial_data.csv"

    # Generate polynomial
    coefficients = generate_polynomial_coefficients(POLY_DEGREE, COEFF_MAX_INTEGER, COEFF_DEC_BITS)
    equation = "y = " + " + ".join([f"{c:.{COEFF_DEC_BITS}f}x^{i}" for i, c in enumerate(coefficients)])

    # Print parameters
    print("▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ POLYNOMIAL DETAILS ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    print(f"Equation: {equation}")
    print("\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ DATA PARAMETERS ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    print(f"X range: 1.90000000-2.10000000 (±0.1 around mean=2)")
    print(f"X precision: 8 decimal places")
    print(f"Y precision: {Y_DECIMAL} decimal places")
    print(f"Sample count: {NUM_SAMPLES} (each with ±0.0005 offsets)")
    print(f"Coefficient range: 0 to {COEFF_MAX_INTEGER} with {COEFF_DEC_BITS} decimal bits")

    # Generate and save data
    dataset = generate_dataset(coefficients, NUM_SAMPLES, Y_DECIMAL)
    save_to_csv(dataset, OUTPUT_FILE, equation, X_DEC_BITS, Y_DECIMAL)
    print(f"\n✅ Data saved to {OUTPUT_FILE}")