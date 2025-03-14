import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import minimize


def load_data_and_coefficients(filename):
    """Load data and extract true coefficients from comment"""
    with open(filename, 'r') as f:
        header = f.readline().strip()
    # Extract coefficients using regex
    matches = re.findall(r"(\d+\.\d+)x\^\d+", header)
    true_coeffs = [float(m) for m in matches]
    return pd.read_csv(filename, comment='#'), true_coeffs


def format_coefficient(c):
    """Format coefficient to 8 decimal places without trailing zeros"""
    return "{:.8f}".format(c).rstrip('0').rstrip('.') if '.' in "{:.8f}".format(c) else "{:.8f}".format(c)


def run_experiment_with_visualization():
    # Load data and true coefficients
    full_data, true_coeffs = load_data_and_coefficients('polynomial_data.csv')

    # Randomly select 10 samples
    np.random.seed(42)
    train_data = full_data.sample(n=10, random_state=42)
    sorted_data = train_data.sort_values('y_base').reset_index(drop=True)

    # Generate fitting x values
    x_fit = np.linspace(1.9, 2.1, len(sorted_data))

    # Add constraints
    constraints = []
    constraint_points = sorted_data.sample(2, random_state=42)
    for _, row in constraint_points.iterrows():
        constraints.append({
            'type': 'eq',
            'fun': lambda c, x=row['x_base'], y=row['y_base']: sum(c[i] * (x ** i) for i in range(len(c))) - y
        })

    # Perform fitting
    try:
        fitted_coeffs = fit_monotonic_poly(sorted_data['y_base'].values, x_fit, constraints)

        # Print coefficients comparison
        print("=== Coefficient Comparison ===")
        print(f"{'Term':<6} {'True':<12} {'Fitted':<12}")
        for i, (tc, fc) in enumerate(zip(true_coeffs, fitted_coeffs)):
            print(f"x^{i:<2} {format_coefficient(tc):<12} {format_coefficient(fc):<12}")

        # Generate curves
        x_plot = np.linspace(1.9, 2.1, 500)
        true_curve = [sum(c * (x ** i) for i, c in enumerate(true_coeffs)) for x in x_plot]
        fitted_curve = [sum(c * (x ** i) for i, c in enumerate(fitted_coeffs)) for x in x_plot]

        # Create visualization
        plt.figure(figsize=(15, 9))

        # Plot original true curve
        plt.plot(x_plot, true_curve, color='#2ca02c', linestyle='--',
                 linewidth=2, label='True Polynomial', alpha=0.7)

        # Plot fitted curve
        plt.plot(x_plot, fitted_curve, color='#ff7f0e', linewidth=2.5,
                 label='Fitted Polynomial')

        # Plot training data with error bars
        plt.errorbar(sorted_data['x_base'], sorted_data['y_base'],
                     yerr=[sorted_data['y_base'] - sorted_data['y_minus'],
                           sorted_data['y_plus'] - sorted_data['y_base']],
                     fmt='o', color='#1f77b4', ecolor='#7f7f7f', markersize=10,
                     elinewidth=1.5, capsize=4, capthick=1.5, zorder=10,
                     label='Training Points (n=10)')

        # Highlight constraint points
        plt.scatter(constraint_points['x_base'], constraint_points['y_base'],
                    s=120, facecolors='none', edgecolors='#d62728',
                    linewidths=2, marker='o', label='Constraint Points (n=2)')

        # Add text annotations
        plt.text(1.905, np.max(true_curve) * 0.95,
                 f"Fitting Accuracy: {calculate_accuracy(fitted_coeffs, train_data) * 100:.1f}%",
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Configure plot
        plt.xlabel('X Value', fontsize=12, labelpad=10)
        plt.ylabel('Y Value', fontsize=12, labelpad=10)
        plt.title('Polynomial Fitting Comparison (True vs Fitted)', fontsize=14, pad=20)
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
        plt.grid(alpha=0.2, linestyle='--')
        plt.xlim(1.895, 2.105)
        plt.tight_layout()

        # Save and show
        plt.savefig('full_comparison.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to full_comparison.png")

    except Exception as e:
        print(f"\nFitting failed: {str(e)}")


def calculate_accuracy(coeffs, data):
    """Calculate prediction accuracy"""
    predictions = data['x_base'].apply(lambda x: sum(c * (x ** i) for i, c in enumerate(coeffs)))
    return ((predictions >= data['y_minus']) & (predictions <= data['y_plus'])).mean()


def fit_monotonic_poly(y_sorted, x_fit, constraints):
    """Optimization core function"""
    degree = 8

    def objective(coeffs):
        return np.mean([(sum(c * (x ** i) for i, c in enumerate(coeffs)) - y) ** 2
                        for x, y in zip(x_fit, y_sorted)])

    # Build constraints
    cons = [{'type': 'ineq',
             'fun': lambda c, x=x: sum((i + 1) * c[i + 1] * (x ** i) for i in range(len(c) - 1))}
            for x in np.linspace(1.9, 2.1, 50)]
    cons += constraints

    # Run optimization
    res = minimize(objective, [1e-3] * (degree + 1),
                   method='SLSQP',
                   bounds=[(1e-12, None)] * (degree + 1),
                   constraints=cons,
                   options={'maxiter': 5000})

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


if __name__ == "__main__":
    run_experiment_with_visualization()