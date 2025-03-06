import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm


def compute_densities(f_expr, x_range=(-3, 3), num_points=500):
    """ Computes and plots the true density, Gaussian approximation, and second-order corrected density for a given f(x)."""

    # Symbolic variable
    x = sp.Symbol('x')

    # Compute derivatives
    grad_f = sp.diff(f_expr, x)
    hess_f = sp.diff(grad_f, x)  # Hessian (second derivative)

    # Check convexity
    convex_check = sp.solve(hess_f < 0, x)  # Solve for where f''(x) < 0
    if convex_check:
        raise ValueError(f"Error: The function {sp.latex(f_expr)} is not convex. Convexity is required.")

    # Solve f'(x) = 0 to find critical points
    critical_points = sp.solve(grad_f, x)

    # Convert functions to numerical
    f_num = sp.lambdify(x, f_expr, 'numpy')
    grad_f_num = sp.lambdify(x, grad_f, 'numpy')
    hess_f_num = sp.lambdify(x, hess_f, 'numpy')

    # Find the minimum numerically
    candidate_mins = []
    for cp in critical_points:
        if hess_f.subs(x, cp) > 0:  # Second derivative test: Hessian must be positive
            candidate_mins.append(float(cp))

    if not candidate_mins:
        raise ValueError("No valid minimum found for the given function.")

    # Choose the global minimum
    x_star = min(candidate_mins, key=lambda cp: f_num(cp))

    # Compute Hessian at x*
    hessian_at_min = hess_f_num(x_star)

    # Define densities
    def true_density(x_val):
        return np.exp(-f_num(x_val)) * np.sqrt(1 + grad_f_num(x_val) ** 2)

    def gaussian_approx(x_val):
        return norm.pdf(x_val, loc=x_star, scale=np.sqrt(1 / hessian_at_min))

    def second_order_approx(x_val):
        gaussian_part = np.exp(-hessian_at_min * (x_val - x_star) ** 2 / 2)
        correction = 1 + (hessian_at_min / 2) * (x_val - x_star) ** 2
        return gaussian_part * correction

    # Compute x values
    x_vals = np.linspace(*x_range, num_points)

    # Compute true density normalization constant
    Z_true, _ = quad(true_density, -np.inf, np.inf)
    true_vals = np.array([true_density(x) for x in x_vals]) / Z_true

    # Compute second-order approximation using its known normalization constant
    Z_2 = np.sqrt(2 * np.pi / hessian_at_min) * 1.5  # Explicitly derived in the paper
    second_order_vals = np.array([second_order_approx(x) for x in x_vals]) / Z_2

    # Compute Gaussian approximation
    gaussian_vals = gaussian_approx(x_vals)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, true_vals, label="True Steady-State Density", linewidth=2, color='orange')
    plt.plot(x_vals, gaussian_vals, label="Gaussian Approximation", linestyle="dashed", linewidth=2, color='red')
    plt.plot(x_vals, second_order_vals, label="Second-Order Approximation", linestyle="dotted", linewidth=2, color='blue')
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(f"Comparison of True Density and Approximations for $f(x)={sp.latex(f_expr)}$\nMinimum at $x^*={x_star:.3f}$")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage: pass any function f(x)=x^2
x = sp.symbols("x")
f_x = x**2+2*x+1

compute_densities(f_x, x_range=(-3, 3))

# A non-convex functino won't work
f_x = sp.symbols('x')**4 - 2*sp.symbols('x')**2  # Example: f(x) = x^4 - 2x^2 -- not convex!
compute_densities(f_x, x_range=(-3, 3))