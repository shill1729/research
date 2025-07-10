"""
This module contains functions for generating toy data.

It generates points uniformly on a curve, with respect to its arc length measure
"""
import numpy as np

from typing import Union
from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.local_dynamics import RiemannianBrownianMotion
from ae.toydata.pointclouds import PointCloud
from ae.toydata.curves import CurveBase
from ae.toydata.surfaces import SurfaceBase
from ae.utils import compute_orthogonal_projection_from_cov


def generate_toy_data(n, surface_or_curve: Union[SurfaceBase, CurveBase], seed=17):
    """
    Generate a Riemannian point cloud of n points and compute the corresponding covariance A matrices.

    :param n:
    :param surface_or_curve:
    :param seed:
    :return:
    """
    bm = RiemannianBrownianMotion()
    manifold = RiemannianManifold(surface_or_curve.local_coords(), surface_or_curve.equation())
    point_cloud = PointCloud(manifold, surface_or_curve.bounds(), bm.drift(manifold), bm.diffusion(manifold), True)
    d = point_cloud.dimension
    print(d)
    x, _, _, cov, _ = point_cloud.generate(n=n, seed=seed)
    # Compute orthogonal projection from covariance and then A matrices.
    p = compute_orthogonal_projection_from_cov(cov, d=d)
    a = np.zeros((n, d + 1, d + 1))
    tangent_length1 = 0.4
    tangent_length2 = 0.3

    for i in range(n):
        normal_length = 5. * np.abs(x[i, 0] * 5 + 0.1)
        eigenvalues, eigenvectors = np.linalg.eigh(p[i])
        if d == 1:
            desired_eigenvalues = np.diag([normal_length ** 2, tangent_length1 ** 2])
        elif d == 2:
            desired_eigenvalues = np.diag([normal_length ** 2, tangent_length1 ** 2, tangent_length2 ** 2])
        else:
            raise NotImplementedError("Only up to intrinsic dimension 2 is implemented")
        a[i] = np.linalg.inv(eigenvectors @ desired_eigenvalues @ eigenvectors.T)
    A_list = [a[i] for i in range(n)]
    return x, A_list, point_cloud


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from ae.toydata.curves import Circle
    num_grid = 50
    xs, A_list, point_cloud = generate_toy_data(5, Circle())

   # Plotting manifold
    curve, _ = point_cloud.get_curve(num_grid)

    # TODO wrap into plot function or method?
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs[:, 0], xs[:, 1], color='black')
    ax.plot(curve[:, 0], curve[:, 1], color="red")
    for i, (x, A) in enumerate(zip(xs, A_list)):
        # Eigen-decompose A_inv to get ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(A)
        # Width and height are 2 * sqrt of eigenvalues (since the ellipse is defined by quadratic form = 1)
        width, height = 2 * np.sqrt(eigvals)
        # Rotation angle in degrees
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ellipse = Ellipse(xy=x, width=width, height=height, angle=angle,
                          edgecolor='blue', facecolor='none', lw=1.5)
        ax.add_patch(ellipse)
    ax.set_aspect('equal')
    ax.set_title("Point Cloud with Tangent-Normal Ellipses")
    plt.show()
