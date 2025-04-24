import matplotlib.pyplot as plt
import numpy as np

def plot_K_surface(K_func, lambda_grid, opt_pts):
    """

    :param K_func:
    :param lambda_grid:
    :param opt_pts:
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("λ1")
    ax.set_ylabel("λ2")
    ax.set_zlabel("K(λ)")
    ax.set_title("Surface Plot of K(λ) over Δ³")
    L1, L2 = np.meshgrid(lambda_grid, lambda_grid, indexing="ij")
    K_vals = np.array([[K_func(np.array([l1, l2, 1 - l1 - l2]))
                        if l1 + l2 <= 1 else np.nan for l2 in lambda_grid] for l1 in lambda_grid])
    ax.plot_surface(L1, L2, K_vals, cmap='inferno', edgecolor='none', alpha=0.2)
    for opt_pt in opt_pts:
        ax.scatter(opt_pt[0], opt_pt[1], opt_pt[2])
    zero_plane = np.zeros_like(K_vals)
    ax.plot_surface(L1, L2, zero_plane, color='gray', alpha=0.4, edgecolor='none')
    return fig