import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tda.solvers.kfunction import get_A_operations_fast, compute_K_fast
from tda.solvers.scipy_solver import minimize_K

def plot_optimal_ellipse_and_centroid(xs, A_list, epsilon2, title, xlim=(-1, 3.5), ylim=(-1.5, 3)):
    # Convert to array
    xs = np.array(xs)
    A_array = np.stack([np.linalg.inv(Ainv) for Ainv in A_list])  # get A from A_inv
    A_inv_array, x_Ainv_x = get_A_operations_fast(A_array, xs)

    # Optimize K(λ)
    result = minimize_K(epsilon2, xs, A_array)
    lmbd_opt = result["lambda"]
    m_lambda = result["m_lambda"]
    A_lambda = result["A_lambda"]
    K_min = result["K_min"]

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'{title}\nOptimal Centroid and Covering Ellipse')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True)

    # Plot original ellipses
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(3):
        A = np.linalg.inv(A_list[i])
        L, V = np.linalg.eigh(A)
        ellipse = np.sqrt(epsilon2) * V @ np.diag(1 / np.sqrt(L)) @ np.array([np.cos(t), np.sin(t)])
        ellipse = ellipse + xs[i].reshape(2, 1)
        ax.plot(ellipse[0], ellipse[1], label=f'Ellipse {i + 1}')

    # Plot optimal centroid
    ax.plot(m_lambda[0], m_lambda[1], 'ko', label='Optimal Centroid $m(\\lambda^*)$')

    # Plot optimal covering ellipse (level set at K(λ*))
    if K_min > 0:
        L, V = np.linalg.eigh(A_lambda)
        theta = np.linspace(0, 2 * np.pi, 200)
        ellipse = np.sqrt(K_min) * V @ np.diag(np.sqrt(L)) @ np.array([np.cos(theta), np.sin(theta)])
        ellipse = ellipse + m_lambda.reshape(2, 1)
        ax.plot(ellipse[0], ellipse[1], 'r--', label='Optimal Covering Ellipse')

    ax.legend()
    return fig

# Define the "all intersecting" case
centers = np.array([(0, 0), (1, 0), (0.5, 1)])
A_list = [
    np.linalg.inv(np.array([[2, 0], [0, 1]])),
    np.linalg.inv(np.array([[1.5, 0.5], [0.5, 1]])),
    np.linalg.inv(np.array([[1, -0.3], [-0.3, 1.2]]))
]
epsilon2 = 1.5

# Plot
fig = plot_optimal_ellipse_and_centroid(centers, A_list, epsilon2, "All Ellipses Intersect")
plt.show()