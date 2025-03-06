import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.pointclouds import PointCloud
from ae.toydata.local_dynamics import BrownianMotion
from ae.toydata.surfaces import Sphere
from ae.utils import compute_orthogonal_projection_from_cov
from tda.solvers.intersection_solver import minimize_K


# -------------------------------
# Plotting Functions
# -------------------------------
def plot_ellipsoidal_ball(ax, center, A, eps, edgecolor='black', lw=2, label=None):
    """
    Plot the ellipse corresponding to the ellipsoidal ball
         { y: (y-center)^T A^{-1} (y-center) <= eps^2 }.
    (2D version)
    """
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    a = eps * np.sqrt(vals[0])
    b = eps * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = plt.Circle(center, a, edgecolor=edgecolor, fill=False, lw=lw, label=label)
    ax.add_patch(ellipse)


def plot_ellipsoid_3d(ax, center, A, eps, color='b', alpha=0.2, wireframe=True):
    """
    Plot a 3D ellipsoid given by (x-center)^T A^{-1} (x-center) <= eps^2.
    Here, A is assumed to be the matrix defining the ellipsoid.
    """
    # Compute eigen-decomposition of A
    eigvals, eigvecs = np.linalg.eigh(A)

    # Create a grid for a unit sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Scale and rotate the sphere to get the ellipsoid.
    scaled_points = points * eps * np.sqrt(eigvals)
    rotated_points = scaled_points @ eigvecs.T
    transformed_points = rotated_points + center
    x_t = transformed_points[:, 0].reshape(x.shape)
    y_t = transformed_points[:, 1].reshape(y.shape)
    z_t = transformed_points[:, 2].reshape(z.shape)

    if wireframe:
        ax.plot_wireframe(x_t, y_t, z_t, color=color, alpha=alpha)
    else:
        ax.plot_surface(x_t, y_t, z_t, color=color, alpha=alpha)


# -------------------------------
# Empirical Timing and Plotting Functions
# -------------------------------
def empirical_intersections_plot_pairs(points, A_list, manifold, surface, eps=0.5, max_plots=6, solver="SLSQP"):
    """
    Loop over every pair of points and their corresponding A matrices,
    time the intersection test (via minimize_K),
    count how many pairs yield a non-empty intersection,
    and plot a few example pair tests.
    """
    n = len(points)
    pair_intersection_count = 0
    pair_total_count = 0
    example_data = []  # to store data for plotting examples

    start_time = time.perf_counter()
    for i, j in itertools.combinations(range(n), 2):
        pair_total_count += 1
        pts = np.array([points[i], points[j]])
        As = [A_list[i], A_list[j]]

        # Put start-time, end-time surrounding this
        start_time = time.perf_counter()
        result = minimize_K(eps, pts, A_list=As, solver=solver)
        end_time = time.perf_counter()
        print("Total minimizer time (including inverting A) = "+str(end_time-start_time))

        # if K_min > 0 then the intersection is non-empty.
        if result['K_min'] > 0:
            pair_intersection_count += 1
        if len(example_data) < max_plots:
            example_data.append((i, j, pts, As, result))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / pair_total_count

    print(f"[Pairs] Total tests: {pair_total_count}, Non-empty intersections: {pair_intersection_count}")
    print(f"[Pairs] Total time: {total_time:.4f} sec, Average time per test: {avg_time:.6f} sec")

    # Plot example pairs
    if example_data:
        n_plots = len(example_data)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, subplot_kw={'projection': '3d'},
                                figsize=(n_cols * 5, n_rows * 5))
        axs = np.array(axs).flatten()
        for ax, (i, j, pts, As, result) in zip(axs, example_data):
            colors = ['red', 'blue']
            # In 3D, we use the provided A (not its inverse) as in your test_surface_intersection.
            _, ax = manifold.plot_manifold_surface(surface.bounds()[0], surface.bounds()[1])
            for idx in range(2):
                plot_ellipsoid_3d(ax, pts[idx], As[idx], eps, color=colors[idx], alpha=0.3)
                ax.scatter(*pts[idx], color=colors[idx], s=50)
            m_lambda = result['m_lambda']
            ax.scatter(*m_lambda, color='black', s=100, marker='x', label='$m_\\lambda$')
            intersect_str = "Non-empty" if result['K_min'] > 0 else "Empty"
            ax.set_title(f"Pair ({i},{j}): {intersect_str}")
            ax.legend()
            # Adjust axis limits for clarity
            max_range = np.max(np.abs(pts)) * 1.5
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
        # Hide any extra axes
        for ax in axs[n_plots:]:
            ax.set_visible(False)
        plt.suptitle("Example Pair Intersection Tests")
        plt.tight_layout()
        plt.show()


def empirical_intersections_plot_triples(points, A_list, manifold, surface, eps=0.5, max_plots=6, solver="SLSQP"):
    """
    Loop over every triple of points and their corresponding A matrices,
    time the intersection test,
    count how many triples yield a non-empty intersection,
    and plot a few example triple tests.
    """
    n = len(points)
    triple_intersection_count = 0
    triple_total_count = 0
    example_data = []

    start_time = time.perf_counter()
    for i, j, k in itertools.combinations(range(n), 3):
        triple_total_count += 1
        pts = np.array([points[i], points[j], points[k]])
        As = [A_list[i], A_list[j], A_list[k]]
        # Put start-time, end-time surrounding this
        start_time = time.perf_counter()
        result = minimize_K(eps, pts, A_list=As, solver=solver)
        end_time = time.perf_counter()

        print("Total minimizer time (includes inverting As) = " + str(end_time - start_time))
        if result['K_min'] > 0:
            triple_intersection_count += 1
        if len(example_data) < max_plots:
            example_data.append((i, j, k, pts, As, result))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / triple_total_count

    print(f"[Triples] Total tests: {triple_total_count}, Non-empty intersections: {triple_intersection_count}")
    print(f"[Triples] Total time: {total_time:.4f} sec, Average time per test: {avg_time:.6f} sec")

    # Plot example triples
    if example_data:
        n_plots = len(example_data)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, subplot_kw={'projection': '3d'},
                                figsize=(n_cols * 5, n_rows * 5))
        axs = np.array(axs).flatten()
        for ax, (i, j, k, pts, As, result) in zip(axs, example_data):
            colors = ['red', 'blue', 'green']
            _, ax = manifold.plot_manifold_surface(surface.bounds()[0], surface.bounds()[1])
            for idx in range(3):
                plot_ellipsoid_3d(ax, pts[idx], As[idx], eps, color=colors[idx], alpha=0.3)
                ax.scatter(*pts[idx], color=colors[idx], s=50)
            m_lambda = result['m_lambda']
            ax.scatter(*m_lambda, color='black', s=100, marker='x', label='$m_\\lambda$')
            intersect_str = "Non-empty" if result['K_min'] > 0 else "Empty"
            ax.set_title(f"Triple ({i},{j},{k}): {intersect_str}")
            ax.legend()
            max_range = np.max(np.abs(pts)) * 1.5
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
        for ax in axs[n_plots:]:
            ax.set_visible(False)
        plt.suptitle("Example Triple Intersection Tests")
        plt.tight_layout()
        plt.show()


def empirical_runtime_on_point_cloud(n=30, eps=0.5, max_plots=6):
    """
    Generate a Riemannian point cloud and the corresponding A matrices,
    then run both the pair and triple empirical intersection tests
    (timing + plotting examples).
    """
    np.random.seed(42)  # For reproducibility
    seed = 17
    # TODO: this has been refactored into 'generate_point_cloud_and_As' in runtime_plots.py. Let's replace this.
    # Generate the point cloud on a sphere
    bm = BrownianMotion()
    sphere = Sphere()
    manifold = RiemannianManifold(sphere.local_coords(), sphere.equation())
    point_cloud = PointCloud(manifold, sphere.bounds(), bm.drift(), bm.diffusion(), True)
    x, _, _, cov, _ = point_cloud.generate(n=n, seed=seed)

    # Compute the orthogonal projections and construct A matrices.
    p = compute_orthogonal_projection_from_cov(cov)
    a = np.zeros((n, 3, 3))
    for i in range(n):
        # We can either do EVD of the orthogonal projection (obtained from SVD of cov) or do EVD of cov.
        eigenvalues, eigenvectors = np.linalg.eigh(p[i])
        s = np.diag(eigenvalues)
        sn = np.zeros((3, 3))
        sn[0, 0] = 1.0
        a[i] = eigenvectors.T @ (s + sn) @ eigenvectors
    A_list = [a[i] for i in range(n)]

    print(f"\n[Point Cloud] n = {n} points, eps = {eps}")
    empirical_intersections_plot_pairs(x, A_list, eps=eps, max_plots=max_plots, manifold=manifold, surface=sphere)
    empirical_intersections_plot_triples(x, A_list, eps=eps, max_plots=max_plots, manifold=manifold, surface=sphere)


if __name__ == '__main__':
    # Run the empirical timing and plotting tests on the Riemannian point cloud.
    empirical_runtime_on_point_cloud(n=10, eps=0.5, max_plots=6)
