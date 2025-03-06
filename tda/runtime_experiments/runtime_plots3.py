import itertools
import time
import numpy as np
import matplotlib.pyplot as plt

from tda.solvers.intersection_solver import minimize_K, cauchy_simplex_minimize_K, projected_gradient_descent
from tda.toydata.pointclouds import generate_point_cloud_and_As

# Import the new solvers
from tda.solvers.our_solver import face_enumeration_solver, active_set_pivoting_solver


def time_intersection_tests(n, eps=0.5, solver="SLSQP", store_results=False):
    """
    For a given number of points n, generate a point cloud and compute the total runtime
    for intersection tests over all pairs and all triples.
    Returns:
      pair_time: total runtime for all pair tests.
      triple_time: total runtime for all triple tests.
      pair_count: number of pairs tested.
      triple_count: number of triples tested.
      (optional) results: A dictionary storing optimal values for comparison.
    """
    x, A_list = generate_point_cloud_and_As(n)

    # Dictionary to store results for comparison
    if store_results:
        pair_results = {}
        triple_results = {}

    # --- Time pairs ---
    pair_count = 0
    start_pairs = time.perf_counter()
    for i, j in itertools.combinations(range(n), 2):
        pts = np.array([x[i], x[j]])
        As = [A_list[i], A_list[j]]

        if solver == "face_enum":
            opt_value = face_enumeration_solver(eps, pts, As)
        elif solver == "active_set":
            opt_value = active_set_pivoting_solver(eps, pts, As)
        elif solver == "CS":
            opt_value = cauchy_simplex_minimize_K(eps, pts, A_list=As)
        elif solver == "pga":
            opt_value = projected_gradient_descent(eps, pts, As)
        else:
            opt_value = minimize_K(eps, pts, A_list=As, solver=solver)
        print(opt_value["lambda"])
        if store_results:
            pair_results[(i, j)] = opt_value["K_min"]

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
            opt_value = face_enumeration_solver(eps, pts, As)
        elif solver == "active_set":
            opt_value = active_set_pivoting_solver(eps, pts, As)
        elif solver == "CS":
            opt_value = cauchy_simplex_minimize_K(eps, pts, A_list=As)
        elif solver == "pga":
            opt_value = projected_gradient_descent(eps, pts, As)
        else:
            opt_value = minimize_K(eps, pts, A_list=As, solver=solver)

        if store_results:
            triple_results[(i, j, k)] = opt_value["K_min"]

        triple_count += 1
    end_triples = time.perf_counter()
    triple_time = end_triples - start_triples

    if store_results:
        return pair_time, triple_time, pair_count, triple_count, pair_results, triple_results
    return pair_time, triple_time, pair_count, triple_count


def compare_solvers(n, eps=0.5, solvers=("SLSQP", "face_enum", "active_set")):
    """
    Runs solvers and compares optimal values across all solvers for consistency.
    """
    print(f"\nComparing solvers for n={n}...")

    # Store results for the first solver
    ref_solver = solvers[0]
    _, _, _, _, ref_pairs, ref_triples = time_intersection_tests(n, eps, ref_solver, store_results=True)

    for solver in solvers[1:]:
        print(f"\nChecking solver: {solver}")
        _, _, _, _, pairs, triples = time_intersection_tests(n, eps, solver, store_results=True)

        # Check pairs
        pair_diff_count = 0
        for key in ref_pairs:
            if key in pairs:
                if not np.isclose(ref_pairs[key], pairs[key], atol=1e-5):
                    print(f"Pair {key}: {ref_solver}={ref_pairs[key]}, {solver}={pairs[key]}")
                    pair_diff_count += 1

        # Check triples
        triple_diff_count = 0
        for key in ref_triples:
            if key in triples:
                if not np.isclose(ref_triples[key], triples[key], atol=1e-5):
                    print(f"Triple {key}: {ref_solver}={ref_triples[key]}, {solver}={triples[key]}")
                    triple_diff_count += 1

        if pair_diff_count == 0 and triple_diff_count == 0:
            print(f"Solver {solver} matches {ref_solver} within tolerance.")
        else:
            print(
                f"Solver {solver} has {pair_diff_count} pair discrepancies and {triple_diff_count} triple discrepancies.")

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
    n_values = [5, 10]

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

    # Compare solvers for correctness
    for n in n_values:
        compare_solvers(n, eps, solvers)

    # Plot runtimes
    plot_runtime_vs_n(n_values, pair_times, triple_times, solvers)


if __name__ == '__main__':
    main()
