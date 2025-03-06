"""
This module contains functions for generating toy data.

It generates points uniformly on a surface, with respect to its volume measure, and
"""
import numpy as np
import sympy as sp

from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.local_dynamics import RiemannianBrownianMotion
from ae.toydata.pointclouds import PointCloud
from ae.toydata.surfaces import Sphere
from ae.utils import compute_orthogonal_projection_from_cov


def generate_point_cloud_and_As(n, seed=17):
    """
    Generate a Riemannian point cloud of n points and compute the corresponding covariance A matrices.

    :param n:
    :param seed:
    :return:
    """
    np.random.seed(42)  # for reproducibility in point sampling
    bm = RiemannianBrownianMotion()
    sphere = Sphere()
    manifold = RiemannianManifold(sphere.local_coords(), sphere.equation())
    point_cloud = PointCloud(manifold, sphere.bounds(), bm.drift(manifold), bm.diffusion(manifold), True)
    x, _, _, cov, _ = point_cloud.generate(n=n, seed=seed)

    # Compute orthogonal projection from covariance and then A matrices.
    p = compute_orthogonal_projection_from_cov(cov)
    a = np.zeros((n, 3, 3))
    for i in range(n):
        eigenvalues, eigenvectors = np.linalg.eigh(p[i])
        s = np.diag(eigenvalues)
        sn = np.zeros((3, 3))
        sn[0, 0] = 2.
        # TODO: make sure these are properly parameterized;
        a[i] = eigenvectors.T @ (s + sn) @ eigenvectors
    A_list = [a[i] for i in range(n)]
    return x, A_list


def compute_precision_matrix(fx, fy, t_sym, t_val):
    """

    :param fx:
    :param fy:
    :param t_sym:
    :param t_val:
    :return:
    """
    p = sp.Matrix([fx, fy])
    p_val = np.array([float(p.subs(t_sym, t_val)[0]), float(p.subs(t_sym, t_val)[1])])
    dp_dt = sp.Matrix([sp.diff(fx, t_sym), sp.diff(fy, t_sym)])
    tangent_sym = dp_dt.subs(t_sym, t_val)
    tangent = np.array([float(tangent_sym[0]), float(tangent_sym[1])])
    if np.linalg.norm(tangent) == 0:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    sigma_t, sigma_n = 0.9 * p_val[0] ** 2, 0.1 * np.sin(p_val[1] * p_val[0]) ** 2 + 0.01
    Q = np.column_stack((tangent, normal))
    D = np.diag([sigma_t, sigma_n])
    A = Q @ D @ Q.T
    return p_val, A
