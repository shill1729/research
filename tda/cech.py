# === Čech Complex Builder ===
import itertools

import numpy as np
from scipy.spatial import cKDTree

from tda.intersection_solver import ellipsoidal_intersection


def build_cech_complex(points, A_list, eps, k_nn=20, max_dim=3):
    """
    Build a Čech complex on the given point cloud.

    Parameters:
      points: numpy array of shape (n, D) with the point cloud.
      A_list: list of n (D x D) positive-definite matrices (one per point) that define the ellipsoidal balls.
      eps: scale parameter ε (each ball is { y: (y - x_i)^T A_i^{-1} (y - x_i) ≤ ε² }).
      k_nn: for each point, only its k_nn nearest neighbors are considered (to shrink search space).
      max_dim: maximum simplex dimension (e.g. max_dim=3 gives vertices, edges, and triangles).

    Returns:
      A list of simplices (each simplex is represented as a tuple of point indices).
    """
    n, D = points.shape
    tree = cKDTree(points)

    # --- Build the 1-skeleton using kNN ---
    one_skeleton = set()
    for i in range(n):
        # Query k_nn neighbors (including self)
        dists, idxs = tree.query(points[i], k=k_nn)
        for j in idxs:
            if j > i:
                xs_pair = points[[i, j], :]
                A_pair = [A_list[i], A_list[j]]
                if ellipsoidal_intersection(eps, xs_pair, A_pair):
                    one_skeleton.add((i, j))

    # Build an undirected graph (as a dictionary of neighbor sets) from the 1-skeleton.
    graph = {i: set() for i in range(n)}
    for (i, j) in one_skeleton:
        graph[i].add(j)
        graph[j].add(i)

    simplices = []
    # All vertices are included.
    for i in range(n):
        simplices.append((i,))
    # Include edges.
    simplices.extend(list(one_skeleton))

    # --- Build higher–dimensional candidate simplices via clique–expansion ---
    # Here we restrict our attention to candidate simplices of size 3 up to (max_dim+1).
    # For each vertex i, we look at combinations of its neighbors (with indices > i) that form a clique.
    for r in range(3, max_dim + 2):  # candidate simplex size r (r vertices -> (r-1)-dim simplex)
        for i in range(n):
            # Consider neighbors of i with index > i (to avoid duplicates)
            neighbors = [j for j in graph[i] if j > i]
            # For each combination of r-1 neighbors:
            for combo in itertools.combinations(neighbors, r - 1):
                candidate = (i,) + combo
                # First, check that every pair in candidate is connected in the 1-skeleton.
                is_clique = True
                for pair in itertools.combinations(candidate, 2):
                    if pair not in one_skeleton and tuple(pair[::-1]) not in one_skeleton:
                        is_clique = False
                        break
                if not is_clique:
                    continue
                # Now, test the ellipsoidal intersection for all points in the candidate.
                xs_candidate = points[list(candidate), :]
                A_candidate = [A_list[j] for j in candidate]
                if ellipsoidal_intersection(eps, xs_candidate, A_candidate):
                    simplices.append(candidate)

    # Remove duplicates and sort (first by dimension, then lexicographically).
    simplices = list(set(simplices))
    simplices.sort(key=lambda s: (len(s), s))
    return simplices


# === Example Test Function ===

def test_cech_complex():
    import matplotlib.pyplot as plt

    # Generate a random 2D point cloud.
    np.random.seed(0)
    n = 50
    points = np.random.rand(n, 2) * 10  # points in [0,10] x [0,10]

    # For illustration, we use the identity as the metric for each point.
    A_list = [np.eye(2) for _ in range(n)]

    # Choose a scale parameter (ball radius) ε.
    eps = 2

    # Build the Čech complex using only the 10 nearest neighbors (to keep the search local)
    cech_complex = build_cech_complex(points, A_list, eps, k_nn=10, max_dim=3)

    print("Čech Complex simplices (vertices, edges, and triangles):")
    for simplex in cech_complex:
        print(simplex)

    # --- Plot the 1-skeleton (vertices and edges) ---
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='black')
    for simplex in cech_complex:
        if len(simplex) == 2:
            i, j = simplex
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'r-', lw=1)
    plt.title("Čech Complex 1-skeleton (ε = {:.2f})".format(eps))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == '__main__':
    test_cech_complex()
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import itertools
import matplotlib.pyplot as plt
from tda.intersection_solver import ellipsoidal_intersection,minimize_K,compute_K

def compute_birth_epsilon(xs, A_list):
    """
    Compute the minimal epsilon where the ellipsoids defined by xs and A_list intersect.
    """
    result = minimize_K(0.0, xs, A_list)
    if not result['success']:
        return np.inf  # Handle optimization failure
    C_max = -result['K_min']
    return np.sqrt(C_max) if C_max > 0 else 0.0

def build_cech_filtration(points, A_list, k_nn=20, max_dim=3):
    """
    Build a filtration of the Čech complex for all candidate simplices up to max_dim.
    Returns a list of (simplex, birth_epsilon) sorted by birth_epsilon and dimension.
    """
    n, D = points.shape
    tree = cKDTree(points)
    filtration = []

    # Add vertices with birth time 0.0
    for i in range(n):
        filtration.append(((i,), 0.0))

    # Generate 1-skeleton candidates using kNN
    edges = set()
    for i in range(n):
        _, idxs = tree.query(points[i], k=k_nn)
        for j in idxs:
            if j != i and (i, j) not in edges and (j, i) not in edges:
                edges.add((i, j))

    # Compute birth epsilon for edges
    for edge in edges:
        i, j = edge
        xs_pair = points[[i, j], :]
        A_pair = [A_list[i], A_list[j]]
        birth_eps = compute_birth_epsilon(xs_pair, A_pair)
        filtration.append((edge, birth_eps))

    # Build adjacency graph for higher-dimensional simplices
    graph = {i: set() for i in range(n)}
    for (i, j) in edges:
        graph[i].add(j)
        graph[j].add(i)

    # Generate higher-dimensional simplices as cliques
    for dim in range(2, max_dim + 1):
        for i in range(n):
            neighbors = [j for j in graph[i] if j > i]
            for combo in itertools.combinations(neighbors, dim):
                candidate = (i,) + combo
                # Check if all subsets are in the graph (clique)
                is_clique = True
                for subset in itertools.combinations(candidate, 2):
                    if subset not in edges and tuple(reversed(subset)) not in edges:
                        is_clique = False
                        break
                if is_clique:
                    xs_candidate = points[list(candidate), :]
                    A_candidate = [A_list[idx] for idx in candidate]
                    birth_eps = compute_birth_epsilon(xs_candidate, A_candidate)
                    filtration.append((candidate, birth_eps))

    # Sort by birth_epsilon and dimension
    filtration.sort(key=lambda x: (x[1], len(x[0])))
    return filtration

def compute_0d_persistence(filtration):
    """
    Compute 0-dimensional persistence pairs using Union-Find.
    """
    parent = {}
    birth_time = {}
    persistence = []

    # Initialize each vertex
    for simplex, birth in filtration:
        if len(simplex) == 1:
            v = simplex[0]
            parent[v] = v
            birth_time[v] = birth

    # Process edges in order
    for simplex, death in filtration:
        if len(simplex) != 2:
            continue
        u, v = simplex
        root_u = find(u, parent)
        root_v = find(v, parent)

        if root_u != root_v:
            if birth_time[root_u] < birth_time[root_v]:
                parent[root_v] = root_u
                persistence.append((birth_time[root_v], death))
            else:
                parent[root_u] = root_v
                persistence.append((birth_time[root_u], death))

    # Add infinite components
    seen = set()
    for simplex, birth in filtration:
        if len(simplex) == 1:
            v = simplex[0]
            root = find(v, parent)
            if root not in seen:
                seen.add(root)
                persistence.append((birth_time[root], np.inf))

    return persistence

def find(u, parent):
    while parent[u] != u:
        parent[u] = parent[parent[u]]  # Path compression
        u = parent[u]
    return u

def plot_barcode(persistence_pairs, title="0-dimensional Barcode"):
    plt.figure(figsize=(8, 4))
    for i, (birth, death) in enumerate(persistence_pairs):
        if death == np.inf:
            death = birth + 1  # Visual offset
        plt.plot([birth, death], [i, i], 'b-', lw=1.5)
    plt.xlabel('Epsilon')
    plt.ylabel('Component')
    plt.title(title)
    plt.show()

def test_cech_complex():
    np.random.seed(0)
    n = 50
    points = np.random.rand(n, 2) * 10
    A_list = [np.eye(2) for _ in range(n)]

    # Build the filtration
    filtration = build_cech_filtration(points, A_list, k_nn=10, max_dim=2)
    print("Filtration computed with", len(filtration), "simplices")

    # Compute 0D persistence
    persistence_0d = compute_0d_persistence(filtration)
    plot_barcode(persistence_0d)

if __name__ == '__main__':
    test_cech_complex()