import numpy as np
import matplotlib.pyplot as plt

from gudhi.persistence_graphical_tools import plot_persistence_diagram, plot_persistence_barcode
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Ellipse


def plot_point_cloud(ax, xs, A_list, curve, eps=1.):
    ax.scatter(xs[:, 0], xs[:, 1], color='black', zorder=3)
    ax.plot(curve[:, 0], curve[:, 1], color='red', lw=1.5, linestyle='--', zorder=2)
    for x, A in zip(xs, A_list):
        eigvals, eigvecs = np.linalg.eigh(A)
        width, height = 2 * np.sqrt(eigvals) * eps
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = Ellipse(xy=x, width=width, height=height, angle=angle,
                          edgecolor='blue', facecolor='none', lw=1.0, zorder=1)
        ax.add_patch(ellipse)


def plot_simplices(ax, xs, edges, triangles=None):
    # Edges as a LineCollection
    edge_segments = [(xs[i], xs[j]) for i, j in edges]
    lc = LineCollection(edge_segments, colors='black', linewidths=1.5, zorder=0)
    ax.add_collection(lc)

    # Optional triangle faces as a PolyCollection
    if triangles:
        polys = [[xs[idx] for idx in tri] for tri in triangles]
        pc = PolyCollection(polys, facecolors='cyan', edgecolors='none', alpha=0.9, zorder=0)
        ax.add_collection(pc)


def plot_diagram_and_barcode(diag, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Persistence diagram (scatters)
    plot_persistence_diagram(diag, axes=ax1)
    ax1.set_title(f'{title}: diagram')

    # Barcode
    plot_persistence_barcode(diag, axes=ax2, max_intervals=1000)
    ax2.set_title(f'{title}: barcode')

    plt.tight_layout()
    plt.show()


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
