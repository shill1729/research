import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection, PolyCollection
from tda.complexes.ellipsoidal_complexes import EllipsoidalVR, EllipsoidalCech
from tda.toydata.curves import generate_curve_point_cloud_and_ellipses


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


def build_and_plot_all(xs, A_list, curve, epsilons):
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    # Define order: VR @ eps[0], Čech @ eps[0], VR @ eps[1], Čech @ eps[1]
    configs = [
        (epsilons[0], EllipsoidalVR, 'VR'),
        (epsilons[0], EllipsoidalCech, 'Čech'),
        (epsilons[1], EllipsoidalVR, 'VR'),
        (epsilons[1], EllipsoidalCech, 'Čech'),
    ]

    for ax, (eps, ComplexClass, name) in zip(axes, configs):
        # Build complex up to triangles
        complex_obj = ComplexClass(xs, A_list, eps)
        complex_obj.build_complex(max_dim=3)

        # Plot base geometry
        plot_point_cloud(ax, xs, A_list, curve, eps)
        plot_simplices(ax, xs, complex_obj.edges, complex_obj.triangles)

        ax.set_title(f"{name} Complex (ε={eps})")
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    xs, A_list, point_cloud = generate_curve_point_cloud_and_ellipses(n=3)
    curve, _ = point_cloud.get_curve(50)
    epsilons = [0.8, 1.25]
    build_and_plot_all(xs, A_list, curve, epsilons)
