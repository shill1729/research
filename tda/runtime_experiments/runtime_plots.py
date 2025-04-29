import itertools
import time
import numpy as np
import matplotlib.pyplot as plt

from tda.solvers.scipy_solver import minimize_K
from tda.solvers.pgd import projected_gradient_descent
from tda.solvers.cauchy_simplex_solver import cauchy_simplex_solver
from tda.toydata.pointclouds import generate_point_cloud_and_pd_matrices


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
    x, A_list = generate_point_cloud_and_pd_matrices(n)

    # --- Time pairs ---
    pair_count = 0
    start_pairs = time.perf_counter()
    for i, j in itertools.combinations(range(n), 2):
        pts = np.array([x[i], x[j]])
        As = np.array([A_list[i], A_list[j]])

        if solver != "cs" and solver != "pga":
            _ = minimize_K(eps, pts, A_array=As, solver=solver)
        elif solver == "cs":
            _ = cauchy_simplex_solver(eps, pts, A_list=As)
        elif solver == "pga":
            _ = projected_gradient_descent(eps, pts, As)
        pair_count += 1
    end_pairs = time.perf_counter()
    pair_time = end_pairs - start_pairs

    # --- Time triples ---
    triple_count = 0
    start_triples = time.perf_counter()
    for i, j, k in itertools.combinations(range(n), 3):
        pts = np.array([x[i], x[j], x[k]])
        As = np.array([A_list[i], A_list[j], A_list[k]])

        if solver != "cs" and solver != "pga":
            _ = minimize_K(eps, pts, A_array=As, solver=solver)
        elif solver == "cs":
            _ = cauchy_simplex_solver(eps, pts, A_list=As)
        elif solver == "pga":
            _ = projected_gradient_descent(eps, pts, As)
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
    eps = 0.01
    # Define a list of n values. Be cautious: triple tests scale as O(n^3)
    n_values = [5, 10, 20, 30, 50, 100]
    # The only relevant ones are SLSQP, trust_constr, and COB-...
    solvers = ["SLSQP"]
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
