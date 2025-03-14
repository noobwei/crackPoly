import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def generate_float_number(integer_bits: int,
                          decimal_bits: int,
                          allow_negative: bool = False) -> float:
    """Generate floating-point number with specified precision"""
    min_int = 10  ** (integer_bits - 1) if integer_bits > 0 else 0
    max_int = (10  ** integer_bits) - 1 if integer_bits > 0 else 0
    integer_part = random.randint(min_int, max_int)
    decimal_part = random.randint(0, (10  ** decimal_bits) - 1) / 10 ** decimal_bits
    number = integer_part + decimal_part
    if allow_negative and random.choice([True, False]):
        number *= -1
    return round(number, decimal_bits)


def generate_polynomial_coefficients(degree: int,
                                     integer_bits: int,
                                     decimal_bits: int) -> List[float]:
    """Generate polynomial coefficients"""
    return [round(generate_float_number(integer_bits, decimal_bits, False), decimal_bits)
            for _ in range(degree + 1)]


def calculate_y(x: float,
                coefficients: List[float],
                y_decimal: int) -> float:
    """Calculate polynomial value"""
    return round(sum(c * (x  ** i) for i, c in enumerate(coefficients)), y_decimal)


def generate_dataset(
        coefficients: List[float],
        num_samples: int,
        x_int_bits: int,
        x_dec_bits: int,
        y_decimal: int,
        allow_negative_x: bool = False
) -> List[Tuple[float, float]]:
    """Generate dataset with full precision values"""
    return [
        (x := generate_float_number(x_int_bits, x_dec_bits, allow_negative_x),
         calculate_y(x, coefficients, y_decimal))
        for _ in range(num_samples)
    ]


def save_to_csv(data: List[Tuple[float, float]],
                filename: str,
                equation: str,
                x_dec: int,
                y_dec: int) -> None:
    """Save data with polynomial equation in header"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Polynomial: {equation}"])
        writer.writerow(['x', 'y'])
        for x, y in data:
            writer.writerow([f"{x:.{x_dec}f}", f"{y:.{y_dec}f}"])


if __name__ == "__main__":
    # Configuration
    POLY_DEGREE = 8
    X_INT_BITS = 1
    X_DEC_BITS = 4
    Y_DECIMAL = 5
    NUM_SAMPLES = 100000
    COEFF_INT_BITS = 2
    COEFF_DEC_BITS = 4
    OUTPUT_FILE = "polynomial_data.csv"

    # Generate polynomial
    coefficients = generate_polynomial_coefficients(POLY_DEGREE, COEFF_INT_BITS, COEFF_DEC_BITS)
    equation = "y = " + " + ".join([f"{c:.{COEFF_DEC_BITS}f}x^{i}" for i, c in enumerate(coefficients)])

    # Print parameters
    print("▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ POLYNOMIAL DETAILS ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    print(f"Equation: {equation}")
    print("\n▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ DATA PARAMETERS ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    print(f"X precision: {X_INT_BITS} integer bits + {X_DEC_BITS} decimal bits")
    print(f"Y precision: {Y_DECIMAL} decimal places")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Coefficient precision: {COEFF_INT_BITS} integer + {COEFF_DEC_BITS} decimal bits")

    # Generate and save data
    dataset = generate_dataset(coefficients, NUM_SAMPLES, X_INT_BITS, X_DEC_BITS, Y_DECIMAL)
    save_to_csv(dataset, OUTPUT_FILE, equation, X_DEC_BITS, Y_DECIMAL)
    print(f"\n✅ Data saved to {OUTPUT_FILE}")

    # Visualization
    plt.figure(figsize=(12, 6))
    x_vals, y_vals = zip(*dataset)

    # Plot data points
    plt.scatter(x_vals, y_vals,
                s=5,  # 点的大小增大到20像素
                alpha=0.4,  # 透明度调整为0.4
                c='#1f77b4',  # 实心蓝色
                edgecolor='none',  # 移除边缘线
                label=f'Data Points (n={NUM_SAMPLES})')

    # # Plot theoretical curve
    # x_min, x_max = min(x_vals), max(x_vals)
    # x_curve = np.linspace(x_min * 0.9, x_max * 1.1, 500)
    # y_curve = [sum(c * (x ** i) for i, c in enumerate(coefficients)) for x in x_curve]
    # plt.plot(x_curve, y_curve, 'r-', lw=2, label='Theoretical Curve')

    # Formatting
    plt.xlabel(f'x Value (precision: {X_INT_BITS}.{X_DEC_BITS} digits)')
    plt.ylabel(f'y Value (precision: {Y_DECIMAL} decimals)')
    plt.title("Polynomial Function Visualization")
    plt.grid(alpha=0.2)
    plt.legend()

    # Save plot
    plt.savefig('polynomial_plot.png', dpi=1000, bbox_inches='tight')
    print("✅ Visualization saved to polynomial_plot.png")