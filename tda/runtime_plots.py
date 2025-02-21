import itertools
import time
import numpy as np
import matplotlib.pyplot as plt

from ae.symbolic.diffgeo import RiemannianManifold
from ae.toydata.pointclouds import PointCloud
from ae.toydata.local_dynamics import RiemannianBrownianMotion
from ae.toydata.surfaces import Sphere
from ae.utils import compute_orthogonal_projection_from_cov
from tda.intersection_solver import minimize_K, cauchy_simplex_minimize_K


def generate_point_cloud_and_As(n, seed=17):
    """
    Generate a Riemannian point cloud of n points and compute the corresponding A matrices.
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
        a[i] = eigenvectors.T @ (s + sn) @ eigenvectors
    A_list = [a[i] for i in range(n)]
    return x, A_list


def time_intersection_tests(n, eps=0.5, solver="SLSQP"):
    """
    For a given number of points n, generate a point cloud and compute the total runtime
    for intersection tests over all pairs and all triples.
    Returns:
      pair_time: total runtime for all pair tests.
      triple_time: total runtime for all triple tests.
      pair_count: number of pairs tested.
      triple_count: number of triples tested.
    """
    x, A_list = generate_point_cloud_and_As(n)

    # --- Time pairs ---
    pair_count = 0
    start_pairs = time.perf_counter()
    for i, j in itertools.combinations(range(n), 2):
        pts = np.array([x[i], x[j]])
        As = [A_list[i], A_list[j]]
        if solver != "CS":
            _ = minimize_K(eps, pts, A_list=As, solver=solver)
        else:
            _ = cauchy_simplex_minimize_K(eps, pts, A_list=As)
        pair_count += 1
    end_pairs = time.perf_counter()
    pair_time = end_pairs - start_pairs

    # --- Time triples ---
    triple_count = 0
    start_triples = time.perf_counter()
    for i, j, k in itertools.combinations(range(n), 3):
        pts = np.array([x[i], x[j], x[k]])
        As = [A_list[i], A_list[j], A_list[k]]
        if solver != "CS":
            _ = minimize_K(eps, pts, A_list=As, solver=solver)
        else:
            _ = cauchy_simplex_minimize_K(eps, pts, A_list=As)
        triple_count += 1
    end_triples = time.perf_counter()
    triple_time = end_triples - start_triples

    return pair_time, triple_time, pair_count, triple_count


def plot_runtime_vs_n(n_values, pair_times, triple_times, solvers, log=False):
    """
    Plot the measured runtime (in seconds) for pairs and triples vs. the number of points.
    """
    plt.figure(figsize=(8, 6))
    for i in range(len(solvers)):
        plt.plot(n_values, pair_times[:, i], 'o-', label='Pairs '+solvers[i])
        plt.plot(n_values, triple_times[:, i], 's-', label='Triples '+solvers[i])
    # plt.plot(n_values, pair_times[:, 1], 'r-', label='Pairs ')
    # plt.plot(n_values, triple_times[:, 1], 'b-', label='Triples ')
    plt.xlabel("Number of Points (n)")
    if log:
        plt.ylabel("Total Runtime (log seconds)")
    else:
        plt.ylabel("Total Runtime (seconds)")
    plt.title("Runtime for Intersection Tests vs. Number of Points")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    eps = 0.5
    # Define a list of n values. Be cautious: triple tests scale as O(n^3)
    n_values = [5, 10, 15, 20]
    # The only relevant ones are SLSQP, trust_constr, and COB-...
    solvers = ["L-BFGS-B","SLSQP", "CS"]
    pair_times = np.zeros((len(n_values), len(solvers)))
    triple_times = np.zeros((len(n_values), len(solvers)))
    print("Measuring runtimes for intersection tests:")
    for j, solver in enumerate(solvers):
        print(solver)
        for i, n in enumerate(n_values):
            pair_time, triple_time, pair_count, triple_count = time_intersection_tests(n, eps, solver)
            print(f"n = {n}: Pairs: {pair_count} tests in {pair_time:.4f} sec; "
                  f"Triples: {triple_count} tests in {triple_time:.4f} sec")
            pair_times[i, j] = pair_time
            triple_times[i, j] = triple_time
    # Plot the measured runtimes vs. the number of points.
    plot_runtime_vs_n(n_values, pair_times, triple_times, solvers)
    # plot_runtime_vs_n(n_values, np.log(pair_times), np.log(triple_times), solvers, log=True)


if __name__ == '__main__':
    main()
