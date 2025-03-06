from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.pointclouds import PointCloud
from ae.toydata.local_dynamics import BrownianMotion
from ae.toydata.surfaces import Sphere
from ae.utils import compute_orthogonal_projection_from_cov
from tda.solvers.intersection_solver import minimize_K

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ellipsoidal_ball(ax, center, A, eps, edgecolor='black', lw=2, label=None):
    """
    Plot the ellipse corresponding to the ellipsoidal ball
         { y: (y-center)^T A^{-1} (y-center) <= eps^2 }.

    We compute the eigen–decomposition of A, and use the fact that if
         A = V diag(d1,d2) V^T,
    then the boundary is given by
         y = center + V @ [eps*sqrt(d1)*cos(t); eps*sqrt(d2)*sin(t)],  t in [0,2pi].
    """
    # Compute eigen-decomposition of A:
    vals, vecs = np.linalg.eigh(A)
    # Sort eigenvalues in descending order so that a is the major semi-axis.
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    a = eps * np.sqrt(vals[0])
    b = eps * np.sqrt(vals[1])
    # Compute the angle (in degrees) of the eigenvector corresponding to the major axis.
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(xy=center, width=2 * a, height=2 * b, angle=angle,
                      edgecolor=edgecolor, fc='none', lw=lw, label=label)
    ax.add_patch(ellipse)


def plot_ellipsoid_3d(ax, center, A, eps, color='b', alpha=0.2, wireframe=True):
    """
    Plot a 3D ellipsoid given by (x-center)^T A^{-1} (x-center) <= eps^2
    """
    # Get eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(A)

    # Create the unit sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Stack the coordinates into a single array
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Transform the sphere into an ellipsoid
    # Scale by eigenvalues
    scaled_points = points * eps * np.sqrt(eigvals)
    # Rotate by eigenvectors
    rotated_points = scaled_points @ eigvecs.T
    # Translate to center
    transformed_points = rotated_points + center

    # Reshape back to grid
    x_t = transformed_points[:, 0].reshape(x.shape)
    y_t = transformed_points[:, 1].reshape(y.shape)
    z_t = transformed_points[:, 2].reshape(z.shape)

    if wireframe:
        ax.plot_wireframe(x_t, y_t, z_t, color=color, alpha=alpha)
    else:
        ax.plot_surface(x_t, y_t, z_t, color=color, alpha=alpha)


# -------------------------------
# Example: 3 points on a circle in the plane
# -------------------------------
def test_three_points_on_circle():
    np.random.seed(42)  # For reproducibility
    k = 3
    d = 2

    # Generate three random angles between 0 and 2*pi and sort them.
    angles = np.sort(2 * np.pi * np.random.rand(k))
    xs = np.zeros((k, d))
    xs[:, 0] = np.cos(angles)
    xs[:, 1] = np.sin(angles)

    # In this example we take A_i = I (the identity) for all i.
    A_list = []
    # Let's define three different positive-definite matrices.
    A_list.append(np.array([[1.0, 0.3],
                            [0.3, 1.2]]))
    A_list.append(np.array([[0.8, -0.2],
                            [-0.2, 1.1]]))
    A_list.append(np.array([[1.3, 0.0],
                            [0.0, 0.9]]))

    # Choose a value of epsilon.
    # (For example, try eps=1.0; depending on the configuration the three unit disks
    #  centered at the three points may or may not have a common intersection.)
    eps = 0.2

    # Minimize K(lambda) over the simplex.
    result = minimize_K(eps, xs, A_list=A_list)

    print("Optimal lambda:", result['lambda'])
    print("Minimum K(lambda):", result['K_min'])
    if result['K_min'] > 0:
        print("=> The intersection of the ellipsoidal balls is non-empty.")
    else:
        print("=> The intersection of the ellipsoidal balls is empty.")

    # Plot the ellipsoidal balls.
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.2, label='Unit circle')
    ax.add_artist(circle)
    colors = ['red', 'green', 'blue']

    for i in range(k):
        center = xs[i]
        plot_ellipsoidal_ball(ax, center, np.linalg.inv(A_list[i]), eps, edgecolor=colors[i],
                              lw=2, label=f'Ellipsoid {i + 1}')
        # Also mark the center.
        ax.plot(center[0], center[1], 'o', color=colors[i])

    # Plot m_lambda from the minimization.
    m_lambda = result['m_lambda']
    ax.plot(m_lambda[0], m_lambda[1], 'ko', label='$m_\\lambda$')

    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title('Three Balls (eps = {:.2f}) and $m_\\lambda$'.format(eps))
    ax.legend()
    plt.show()


def test_riemannian_manifold():
    np.random.seed(42)  # For reproducibility
    seed = 17
    num_pts = 30
    bm = BrownianMotion()
    sphere = Sphere()
    manifold = RiemannianManifold(sphere.local_coords(), sphere.equation())
    point_cloud = PointCloud(manifold, sphere.bounds(), bm.drift(), bm.diffusion(), True)
    # Sample uniformly:
    x, _, _, cov, _ = point_cloud.generate(n=num_pts, seed=seed)
    p = compute_orthogonal_projection_from_cov(cov)
    a = np.zeros((num_pts, 3, 3))
    for i in range(num_pts):
        eigenvalues, eigenvectors = np.linalg.eigh(p[i])
        s = np.diag(eigenvalues)
        sn = np.zeros((3, 3))
        sn[0, 0] = 1.
        a[i] = eigenvectors.T @ (s + sn) @ eigenvectors
    A_list = [a[i] for i in range(num_pts)]

    # Choose a value of epsilon.
    # (For example, try eps=1.0; depending on the configuration the three unit disks
    #  centered at the three points may or may not have a common intersection.)
    eps = 0.5

    # Take first k points and their corresponding A matrices
    k = 2
    points = x[:k]
    A_matrices = A_list[:k]
    # Test the intersection
    fig, ax = manifold.plot_manifold_surface(sphere.bounds()[0], sphere.bounds()[1])
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    result = test_surface_intersection(points, A_matrices, eps=eps, ax=ax)
    plt.show()
    return result


def test_surface_intersection(surface_pts, A_matrices=None, eps=1.0, ax=None):
    """
    Test intersection of ellipsoids on a 3D surface

    Parameters:
    surface_pts : array of shape (k, 3) containing points on the surface
    A_matrices : list of k (3x3) positive definite matrices
    eps : radius parameter
    """
    k = len(surface_pts)
    if A_matrices is None:
        A_matrices = [np.eye(3) for _ in range(k)]

    result = minimize_K(eps, surface_pts, A_list=A_matrices)

    print("Optimal lambda:", result['lambda'])
    print("Minimum K(lambda):", result['K_min'])
    if result['K_min'] > 0:
        print("=> The intersection of the ellipsoidal balls is non-empty.")
    else:
        print("=> The intersection of the ellipsoidal balls is empty.")

    if ax is None:
        # Visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i in range(k):
        plot_ellipsoid_3d(ax, surface_pts[i], A_matrices[i], eps,
                          color=colors[i % len(colors)], alpha=0.2)
        ax.scatter(*surface_pts[i], color=colors[i % len(colors)], s=100)

    # Plot the optimal point
    m_lambda = result['m_lambda']

    ax.scatter(*m_lambda, color='black', s=100, label='$m_\\lambda$')

    # Set reasonable axis limits
    max_range = np.max(np.abs(surface_pts)) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ellipsoid Intersection (eps = {eps:.2f})')
    ax.legend()

    # plt.show()

    return result


# Example usage:
if __name__ == '__main__':
    test_three_points_on_circle()
    test_riemannian_manifold()

