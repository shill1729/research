import itertools
import time
import numpy as np
import matplotlib.pyplot as plt

from tda.solvers.intersection_solver import minimize_K, cauchy_simplex_minimize_K, projected_gradient_descent
from tda.toydata.pointclouds import generate_point_cloud_and_As

# Import the new solvers
from tda.solvers.our_solver import face_enumeration_solver, active_set_pivoting_solver


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

        if solver == "face_enum":
            _ = face_enumeration_solver(eps, pts, As)
        elif solver == "active_set":
            _ = active_set_pivoting_solver(eps, pts, As)
        elif solver == "CS":
            _ = cauchy_simplex_minimize_K(eps, pts, A_list=As)
        elif solver == "pga":
            _ = projected_gradient_descent(eps, pts, As)
        else:
            _ = minimize_K(eps, pts, A_list=As, solver=solver)

        pair_count += 1
    end_pairs = time.perf_counter()
    pair_time = end_pairs - start_pairs

    # --- Time triples ---
    triple_count = 0
    start_triples = time.perf_counter()
    for i, j, k in itertools.combinations(range(n), 3):
        pts = np.array([x[i], x[j], x[k]])
        As = [A_list[i], A_list[j], A_list[k]]

        if solver == "face_enum":
            _ = face_enumeration_solver(eps, pts, As)
        elif solver == "active_set":
            _ = active_set_pivoting_solver(eps, pts, As)
        elif solver == "CS":
            _ = cauchy_simplex_minimize_K(eps, pts, A_list=As)
        elif solver == "pga":
            _ = projected_gradient_descent(eps, pts, As)
        else:
            _ = minimize_K(eps, pts, A_list=As, solver=solver)

        triple_count += 1
    end_triples = time.perf_counter()
    triple_time = end_triples - start_triples

    return pair_time, triple_time, pair_count, triple_count


def plot_runtime_vs_n(n_values, pair_times, triple_times, solvers, log=False):
    """
    Plot the measured runtime (in seconds) for pairs and triples vs. the number of points.
    - Uses consistent colors per solver.
    - Pairs: Dashed lines.
    - Triples: Solid lines.
    """
    plt.figure(figsize=(8, 6))

    # Use the updated Matplotlib colormap API
    colormap = plt.colormaps["tab10"]  # Corrected for Matplotlib 3.7+

    for i, solver in enumerate(solvers):
        color = colormap(i)  # Get color for this solver
        plt.plot(n_values, pair_times[:, i], 'o--', color=color, label=f'Pairs {solver}')  # Dashed for pairs
        plt.plot(n_values, triple_times[:, i], 's-', color=color, label=f'Triples {solver}')  # Solid for triples

    plt.xlabel("Number of Points (n)")
    plt.ylabel("Total Runtime (log seconds)" if log else "Total Runtime (seconds)")
    plt.title("Runtime for Intersection Tests vs. Number of Points")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    eps = 0.5
    # Define a list of n values. Be cautious: triple tests scale as O(n^3)
    n_values = [5, 10, 20, 30, 40, 50]

    # Add the new solvers for benchmarking
    solvers = ["SLSQP", "face_enum", "active_set"]

    pair_times = np.zeros((len(n_values), len(solvers)))
    triple_times = np.zeros((len(n_values), len(solvers)))

    print("Measuring runtimes for intersection tests:")

    for j, solver in enumerate(solvers):
        print(f"Testing solver: {solver}")
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
